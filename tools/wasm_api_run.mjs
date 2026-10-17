// Runs the engine's API (docs/WASM-API.md) as WebAssembly under Node, through the app's loader
// (crates/wasm/js/engine.mjs), for tools/wasm_check.sh and the speed measurements.
//
//   node tools/wasm_api_run.mjs <module.wasm> [--equity F] [--net F]... eval REQ.json...
//        one result per line (as `engine-native eval` prints them)
//   node tools/wasm_api_run.mjs <module.wasm> [--equity F] [--net F]... shards K REQ.json...
//        each request split into K shards, valued one by one and merged: one result per line
//   node tools/wasm_api_run.mjs <module.wasm> [--equity F] [--net F]... [--repeat R] time REQ.json...
//        milliseconds per call and per deal (the best of R runs); load times and memory on stderr
import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { load } from "../crates/wasm/js/engine.mjs";

const USAGE =
  "usage: node tools/wasm_api_run.mjs <module.wasm> [--equity F] [--net F]... [--repeat R] (eval|shards K|time) REQ.json...";

function fail(message) {
  console.error(message);
  process.exit(2);
}

const [path, ...rest] = process.argv.slice(2);
if (!path) fail(USAGE);
const t0 = performance.now();
const engine = await load(readFileSync(path));
console.error(`module ${path}: instantiated in ${(performance.now() - t0).toFixed(0)} ms`);

let repeat = 1;
const args = [];
for (let i = 0; i < rest.length; i++) {
  const a = rest[i];
  if (a === "--equity" || a === "--net" || a === "--student") {
    const f = rest[++i] ?? fail(`${a} needs a file`);
    const bytes = readFileSync(f);
    const t = performance.now();
    const j =
      a === "--equity" ? engine.loadEquity(bytes) : a === "--student" ? engine.loadStudent(bytes) : engine.loadNet(bytes);
    console.error(`loaded ${f} in ${(performance.now() - t).toFixed(0)} ms: ${JSON.stringify(j)}`);
  } else if (a === "--repeat") {
    repeat = Number(rest[++i]);
  } else {
    args.push(a);
  }
}
console.error(`memory after loading: ${engine.info().memoryBytes} bytes`);

const [cmd, ...files] = args;
if (cmd === "eval") {
  for (const f of files) console.log(engine.evaluate(readFileSync(f, "utf8")));
} else if (cmd === "shards") {
  const k = Number(files.shift());
  for (const f of files) {
    const req = JSON.parse(readFileSync(f, "utf8"));
    // A student request is one call (no deals to split): it is merged alone.
    const shards = req.method === "student" ? 1 : Math.min(k, req.deals);
    const parts = [];
    for (let i = shards - 1; i >= 0; i--) parts.push(engine.evaluate({ ...req, shard: i, shards }));
    console.log(engine.merge(parts));
  }
} else if (cmd === "time") {
  for (const f of files) {
    const text = readFileSync(f, "utf8");
    let best = Infinity;
    let out = "";
    for (let r = 0; r < repeat; r++) {
      const t = performance.now();
      out = engine.evaluate(text);
      best = Math.min(best, performance.now() - t);
    }
    const v = JSON.parse(out);
    const deals = (v.dealsTo ?? 0) - (v.dealsFrom ?? 0);
    console.log(
      `${f}\tstatus=${v.status}\tmethod=${v.method ?? v.code}\ttrick=${v.trick}\tdeals=${deals}\tms=${best.toFixed(1)}\tms_per_deal=${deals > 0 ? (best / deals).toFixed(2) : 0}`,
    );
  }
  console.error(`memory after the runs: ${engine.info().memoryBytes} bytes`);
} else {
  fail(USAGE);
}
