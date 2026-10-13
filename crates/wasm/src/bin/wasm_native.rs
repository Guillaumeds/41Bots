//! `wasm-native <seed> <matches>`: prints `result_hash(seed, matches)` in decimal, computed
//! natively. `tools/wasm_check.sh` compares it with the same function run as WebAssembly.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let parsed = match args.as_slice() {
        [seed, matches] => seed.parse::<u64>().ok().zip(matches.parse::<u32>().ok()),
        _ => None,
    };
    let Some((seed, matches)) = parsed else {
        eprintln!("usage: wasm-native <seed: 0..2^64> <matches: 0..2^32>");
        return ExitCode::from(2);
    };
    println!("{}", wasm::result_hash(seed, matches));
    ExitCode::SUCCESS
}
