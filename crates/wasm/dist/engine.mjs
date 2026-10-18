// The 41 engine's WebAssembly API (docs/WASM-API.md): a tiny loader with no dependencies, for a
// browser page, a Web Worker, Node 18+ or a Cloudflare Worker.
//
//   import { load } from "./engine.mjs";
//   const engine = await load(wasmBytesOrModule);
//   engine.loadEquity(equityBytes, "64740dab…");   // the SHA-256 is checked when given
//   engine.loadNet(o3Bytes, "f5766aa9…");          // -> { sha256, trick, bytes }
//   engine.loadStudent(studentBytes);               // -> { sha256, bytes } (requests with method "student")
//   const result = engine.evaluate(request);       // request: object or JSON text -> JSON text
//   const merged = engine.merge([r0, r1]);         // results: objects or JSON texts -> JSON text
//   engine.info();                                 // -> { engine, api, valuesVersion, equity, nets, memoryBytes }
//
// Bytes cross into the module through a buffer the module owns: `inbox(len)` returns its address
// in the module's memory, the bytes are written there, the operation runs, and its output is read
// from `outbox()` / `outbox_len()`. Every call but `load` is synchronous: run `evaluate` in a Web
// Worker. If the module ever traps (an engine bug), this engine object refuses every later call
// (its message starts with `engine_dead`): load a new one.

const encoder = new TextEncoder();
const decoder = new TextDecoder();

/** The API version this loader speaks. */
export const API_VERSION = 1;

function toBytes(x, what) {
  if (x instanceof Uint8Array) return x;
  if (x instanceof ArrayBuffer) return new Uint8Array(x);
  if (ArrayBuffer.isView(x)) return new Uint8Array(x.buffer, x.byteOffset, x.byteLength);
  throw new TypeError(`${what} must be an ArrayBuffer or a typed array`);
}

/** Instantiate the module (bytes or a compiled WebAssembly.Module) and return an Engine. */
export async function load(source) {
  const module =
    source instanceof WebAssembly.Module ? source : await WebAssembly.compile(toBytes(source, "the module"));
  const instance = await WebAssembly.instantiate(module, {});
  return new Engine(instance);
}

export class Engine {
  #x;
  #dead = null;

  constructor(instance) {
    this.#x = instance.exports;
    const v = this.#x.api_version();
    if (v !== API_VERSION) throw new Error(`api_mismatch: the module speaks API ${v}, this loader ${API_VERSION}`);
  }

  /** Run export `name` on `bytes` (or with no input when `bytes` is null); [status, output text]. */
  #call(name, bytes) {
    if (this.#dead) throw new Error(`engine_dead: ${this.#dead}`);
    const x = this.#x;
    try {
      let status;
      if (bytes === null) {
        status = x[name]();
      } else {
        const at = x.inbox(bytes.length);
        new Uint8Array(x.memory.buffer, at, bytes.length).set(bytes);
        status = x[name](bytes.length);
      }
      const out = new Uint8Array(x.memory.buffer, x.outbox(), x.outbox_len());
      return [status, decoder.decode(out)];
    } catch (e) {
      this.#dead = `the module trapped (${e && e.message ? e.message : e}); load a new engine`;
      throw new Error(`engine_dead: ${this.#dead}`);
    }
  }

  #load(name, bytes, sha256) {
    const data = toBytes(bytes, "the file");
    if (sha256 !== undefined && sha256 !== null) {
      if (!/^[0-9a-f]{64}$/.test(sha256)) throw new Error("bad_request: the SHA-256 must be 64 lowercase hex digits");
      this.#call("expect_sha256", encoder.encode(sha256));
    }
    const [status, text] = this.#call(name, data);
    const j = JSON.parse(text);
    if (status !== 0) throw new Error(`${j.code}: ${j.reason}`);
    return j;
  }

  /** Load the equity table (.npy bytes). Networks loaded before are dropped. -> { sha256, bytes } */
  loadEquity(bytes, sha256) {
    return this.#load("load_equity", bytes, sha256);
  }

  /** Load an oracle network (.safetensors bytes); its trick comes from the file. -> { sha256, trick, bytes } */
  loadNet(bytes, sha256) {
    return this.#load("load_net", bytes, sha256);
  }

  /** Load the student network (.safetensors bytes): requests with `method: "student"` at tricks 1-2. -> { sha256, bytes } */
  loadStudent(bytes, sha256) {
    return this.#load("load_student", bytes, sha256);
  }

  /** Value a decision: the request (object or JSON text) -> the result or refusal (JSON text). */
  evaluate(request) {
    const text = typeof request === "string" ? request : JSON.stringify(request);
    return this.#call("evaluate", encoder.encode(text))[1];
  }

  /** Merge the shards of one search (objects or JSON texts) -> the whole search's result (JSON text). */
  merge(results) {
    if (!Array.isArray(results)) throw new TypeError("merge takes an array of results");
    const text = `[${results.map((r) => (typeof r === "string" ? r : JSON.stringify(r))).join(",")}]`;
    return this.#call("merge", encoder.encode(text))[1];
  }

  /** The engine's version, the loaded files and the module's memory in bytes. */
  info() {
    return JSON.parse(this.#call("info", null)[1]);
  }
}
