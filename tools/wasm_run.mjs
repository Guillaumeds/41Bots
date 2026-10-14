// Runs the 41 engine's WebAssembly module under Node, for tools/wasm_check.sh.
//
//   node tools/wasm_run.mjs <module.wasm> --describe        list the module's imports and exports
//   node tools/wasm_run.mjs <module.wasm> <seed> <matches>  print result_hash(seed, matches)
//
// The module is instantiated with an empty import object: the engine needs nothing from the host,
// so any import would make instantiation fail. u64 values cross the boundary as BigInt (wasm i64);
// the result comes back signed and is printed as the unsigned decimal the native binary prints.
import { readFileSync } from "node:fs";

const USAGE = "usage: node tools/wasm_run.mjs <module.wasm> (--describe | <seed> <matches>)";

function fail(message) {
  console.error(message);
  process.exit(2);
}

function parseUnsigned(text, bits, what) {
  if (!/^[0-9]+$/.test(text ?? "")) fail(`${what} must be a whole number: ${text}\n${USAGE}`);
  const value = BigInt(text);
  if (value >= 1n << BigInt(bits)) fail(`${what} must be below 2^${bits}: ${text}`);
  return value;
}

const [path, ...rest] = process.argv.slice(2);
const describe = rest.length === 1 && rest[0] === "--describe";
if (!path || (!describe && rest.length !== 2)) fail(USAGE);

const module = await WebAssembly.compile(readFileSync(path));

if (describe) {
  const list = (items) =>
    items.length === 0
      ? "none"
      : items.map((d) => `${d.module ? `${d.module}.` : ""}${d.name} (${d.kind})`).join(", ");
  console.log(`imports: ${list(WebAssembly.Module.imports(module))}`);
  console.log(`exports: ${list(WebAssembly.Module.exports(module))}`);
} else {
  const seed = parseUnsigned(rest[0], 64, "seed");
  const matches = Number(parseUnsigned(rest[1], 32, "matches"));
  const instance = await WebAssembly.instantiate(module, {});
  const hash = BigInt.asUintN(64, instance.exports.result_hash(seed, matches));
  console.log(hash.toString());
}
