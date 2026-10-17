// Splits one request of the engine's API (docs/WASM-API.md §6) over K Node worker threads, each
// with its own instance of the module, and merges the shards: prints the merged result on stdout
// and the wall time of the search (after loading) on stderr. The app does the same with Web
// Workers.
//
//   node tools/wasm_workers.mjs <module.wasm> --equity F --net F --workers K REQ.json
import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { load } from "../crates/wasm/js/engine.mjs";

if (isMainThread) {
  const [path, ...rest] = process.argv.slice(2);
  const opt = { nets: [] };
  for (let i = 0; i < rest.length; i++) {
    if (rest[i] === "--equity") opt.equity = rest[++i];
    else if (rest[i] === "--net") opt.nets.push(rest[++i]);
    else if (rest[i] === "--workers") opt.workers = Number(rest[++i]);
    else opt.request = rest[i];
  }
  const req = JSON.parse(readFileSync(opt.request, "utf8"));
  const k = Math.min(opt.workers, req.deals);
  const module = await WebAssembly.compile(readFileSync(path));
  const workers = [];
  for (let i = 0; i < k; i++) {
    workers.push(new Worker(new URL(import.meta.url), { workerData: { module, equity: opt.equity, nets: opt.nets } }));
  }
  const ready = (w) => new Promise((ok) => w.once("message", ok));
  const loads = await Promise.all(workers.map(ready));
  console.error(`${k} workers loaded in ${Math.max(...loads.map((l) => l.ms)).toFixed(0)} ms (slowest); memory ${loads[0].memory} bytes each`);
  const t = performance.now();
  const parts = await Promise.all(
    workers.map((w, i) => {
      const p = ready(w);
      w.postMessage({ ...req, shard: i, shards: k });
      return p;
    }),
  );
  const ms = performance.now() - t;
  const engine = await load(module);
  console.log(engine.merge(parts.map((p) => p.result)));
  console.error(`search: ${ms.toFixed(0)} ms wall for ${req.deals} deals on ${k} workers (${(ms / req.deals).toFixed(1)} ms per deal)`);
  for (const w of workers) await w.terminate();
} else {
  const t = performance.now();
  const engine = await load(workerData.module);
  engine.loadEquity(readFileSync(workerData.equity));
  for (const n of workerData.nets) engine.loadNet(readFileSync(n));
  parentPort.postMessage({ ms: performance.now() - t, memory: engine.info().memoryBytes });
  parentPort.on("message", (req) => parentPort.postMessage({ result: engine.evaluate(req) }));
}
