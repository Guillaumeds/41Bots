//! `engine-native`: the app's API (docs/WASM-API.md) run natively, through the same exports the
//! WebAssembly module has, for `tools/wasm_check.sh` (native and WebAssembly results must be the
//! same bytes) and the speed measurements.
//!
//! ```text
//! engine-native [--equity FILE] [--net FILE]... eval REQUEST.json...    one result per line
//! engine-native merge RESULTS.json                                      a JSON array of results
//! engine-native time [--equity F] [--net F]... [--repeat R] REQUEST.json...
//! engine-native fixtures DIR                                            write the check's requests
//! engine-native bench DIR [DEALS [PER_PLACE]]                           write the speed requests
//! engine-native info [--equity F] [--net F]...
//! ```

use std::process::ExitCode;
use std::time::Instant;

use wasm::ffi;

fn load(flag: &str, path: &str) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let n = ffi::set_input(&bytes);
    let status = match flag {
        "--equity" => ffi::load_equity(n),
        _ => ffi::load_net(n),
    };
    let out = String::from_utf8(ffi::output()).unwrap_or_default();
    if status != 0 {
        return Err(format!("{path}: {out}"));
    }
    eprintln!("loaded {path}: {out}");
    Ok(())
}

fn evaluate(path: &str) -> Result<String, String> {
    let req = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let n = ffi::set_input(&req);
    ffi::evaluate(n);
    String::from_utf8(ffi::output()).map_err(|e| e.to_string())
}

fn run(args: &[String]) -> Result<(), String> {
    let mut rest = Vec::new();
    let mut repeat = 1usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            f @ ("--equity" | "--net") => {
                let p = args.get(i + 1).ok_or(format!("{f} needs a file"))?;
                load(f, p)?;
                i += 2;
            }
            "--repeat" => {
                repeat = args
                    .get(i + 1)
                    .and_then(|r| r.parse().ok())
                    .ok_or("--repeat needs a number")?;
                i += 2;
            }
            _ => {
                rest.push(args[i].clone());
                i += 1;
            }
        }
    }
    let Some((cmd, files)) = rest.split_first() else {
        return Err(
            "usage: engine-native [--equity F] [--net F]... (eval|merge|time|fixtures|info) ..."
                .into(),
        );
    };
    match cmd.as_str() {
        "eval" => {
            for f in files {
                println!("{}", evaluate(f)?);
            }
        }
        "merge" => {
            let f = files.first().ok_or("merge needs a file")?;
            let bytes = std::fs::read(f).map_err(|e| format!("{f}: {e}"))?;
            let n = ffi::set_input(&bytes);
            ffi::merge(n);
            println!(
                "{}",
                String::from_utf8(ffi::output()).map_err(|e| e.to_string())?
            );
        }
        "time" => {
            for f in files {
                let mut best = f64::INFINITY;
                let mut out = String::new();
                for _ in 0..repeat {
                    let t = Instant::now();
                    out = evaluate(f)?;
                    best = best.min(t.elapsed().as_secs_f64() * 1e3);
                }
                let v: serde_json::Value = serde_json::from_str(&out).map_err(|e| e.to_string())?;
                let deals =
                    v["dealsTo"].as_u64().unwrap_or(0) - v["dealsFrom"].as_u64().unwrap_or(0);
                println!(
                    "{f}\tstatus={}\tmethod={}\ttrick={}\tdeals={deals}\tms={best:.1}\tms_per_deal={:.2}",
                    v["status"].as_str().unwrap_or("?"),
                    v["method"]
                        .as_str()
                        .unwrap_or(v["code"].as_str().unwrap_or("?")),
                    v["trick"],
                    if deals > 0 { best / deals as f64 } else { 0.0 }
                );
            }
        }
        "fixtures" | "bench" => {
            let dir = files.first().ok_or("fixtures needs a directory")?;
            std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
            let list = if cmd == "bench" {
                let deals = files.get(1).and_then(|d| d.parse().ok()).unwrap_or(8);
                let per = files.get(2).and_then(|d| d.parse().ok()).unwrap_or(4);
                wasm::fixtures::bench_requests(deals, per)
            } else {
                wasm::fixtures::requests()
            };
            for (name, req) in list {
                let p = format!("{dir}/{name}.json");
                std::fs::write(&p, format!("{req}\n")).map_err(|e| format!("{p}: {e}"))?;
                eprintln!("wrote {p}");
            }
        }
        "info" => {
            ffi::info();
            println!(
                "{}",
                String::from_utf8(ffi::output()).map_err(|e| e.to_string())?
            );
        }
        c => return Err(format!("unknown command {c}")),
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("engine-native: {e}");
            ExitCode::from(2)
        }
    }
}
