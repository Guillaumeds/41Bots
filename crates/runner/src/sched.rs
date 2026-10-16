//! Format-2 manifests and the scheduling rules (docs/M3-SPEC.md sections 7.1 and 7.2), as pure
//! functions over the queue state and a clock value, so they are unit-tested with a fake clock.
//!
//! Manifest fields (section 7.1's example, plus the optional ones marked *):
//! - `format: 2`, `run_id`, `run_seed`, `deadline_utc` (optional), `threads` (null = the usable
//!   cores), `bin_sha` (checked against the machine's `BIN_SHA`), `inputs` (`name:
//!   "sha256:<hex>"`, fetched from `inputs/sha256/<hex>` and verified), `upload_every_s` (≤ 600,
//!   default 600: the done-list and the running chunks' live files), `heartbeat_every_s` (default
//!   30), *`benchmark_seconds` (start-up benchmark,
//!   default 60, 0 = none);
//! - per stream: `name`, `priority` (lower first), `kind`, `params`, `chunks` (a number, or
//!   `"until_decided"` for an `arena_blocks` stream with `params.sprt`);
//! - stream params read by the scheduler: *`threads` (per chunk, default 1, or `"all"`),
//!   *`chunk_sizes` (one per chunk: longest first), *`est_seconds` (the duration assumed by the
//!   deadline rule until a chunk of the kind has finished), `blocks_per_chunk`, `sprt` (`s0`,
//!   `s1`, `alpha`, `beta`, `max_blocks`, *`reject_only`); the rest goes to the chunk kind.
//! - *`machine` (top level): what the launcher rents (`python/launcher/launch.py`, e.g.
//!   `{"gpus": 1, "min_gpu_ram_gb": 4, "min_cores": 128, "disk_gb": 80}`); the runner ignores it
//!   (it is kept in the final manifest with the rest).

use std::collections::{BTreeMap, BTreeSet};

use arena::sprt::{Decision, Moments, SprtParams, decide, llr};
use engine::rng::Rng;
use serde_json::{Value, json};

/// Kinds a format-2 stream may run.
pub const KINDS: [&str; 12] = [
    "difftest",
    "equity_sweep",
    "benchmark",
    "solver_spike",
    "search_measure",
    "t1bench",
    "teacher",
    "arena_blocks",
    "suitability",
    "agree",
    "belief_bench",
    "hybrid_rung",
];

/// The keys of a stream object; everything else goes in `params`.
pub const STREAM_KEYS: [&str; 5] = ["name", "priority", "kind", "params", "chunks"];

/// Stream params the scheduler reads (the rest goes to the chunk kind): a stream-level copy of
/// one of these is a misplaced param.
const PARAMS_KEYS: [&str; 5] = [
    "threads",
    "chunk_sizes",
    "est_seconds",
    "blocks_per_chunk",
    "sprt",
];

/// How many chunks a stream has.
#[derive(Clone, Debug, PartialEq)]
pub enum Chunks {
    Fixed(u64),
    /// SPRT streams: chunks until the test over the finished prefix decides (or the cap).
    UntilDecided,
}

/// The sequential test of an `arena_blocks` stream.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamSprt {
    pub params: SprtParams,
    pub max_blocks: u64,
    /// Stage 1 of a two-stage test (PLAN A9): it can only reject.
    pub reject_only: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stream {
    pub name: String,
    pub priority: u64,
    pub kind: String,
    pub params: Value,
    pub chunks: Chunks,
    /// Threads one chunk uses (`params.threads`, default 1; `"all"` is the whole machine, see
    /// [`Manifest2::resolve_threads`]).
    pub threads: usize,
    pub threads_all: bool,
    /// Known chunk sizes (`params.chunk_sizes`), for longest-first order.
    pub sizes: Option<Vec<u64>>,
    /// Duration guess before any chunk of this kind has finished (`params.est_seconds`).
    pub est_seconds: f64,
    /// Blocks per chunk (`arena_blocks` only).
    pub blocks_per_chunk: u64,
    pub sprt: Option<StreamSprt>,
}

impl Stream {
    /// Total chunks this stream can ever issue.
    pub fn total(&self) -> u64 {
        match self.chunks {
            Chunks::Fixed(n) => n,
            Chunks::UntilDecided => self
                .sprt
                .as_ref()
                .map_or(0, |s| s.max_blocks.div_ceil(self.blocks_per_chunk.max(1))),
        }
    }

    /// Chunk indices in issue order: longest first when sizes are known, else index order.
    pub fn order(&self) -> Vec<u64> {
        let mut v: Vec<u64> = (0..self.total()).collect();
        if let Some(s) = &self.sizes {
            v.sort_by_key(|&i| {
                (
                    std::cmp::Reverse(s.get(i as usize).copied().unwrap_or(0)),
                    i,
                )
            });
        }
        v
    }

    /// Blocks that chunk `i` plays (`arena_blocks`).
    pub fn blocks_in(&self, i: u64) -> u64 {
        let bpc = self.blocks_per_chunk.max(1);
        let cap = self.sprt.as_ref().map_or(u64::MAX, |s| s.max_blocks);
        bpc.min(cap.saturating_sub(i * bpc))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Manifest2 {
    pub run_id: String,
    pub run_seed: u64,
    pub deadline: Option<u64>,
    pub threads: Option<usize>,
    pub bin_sha: Option<String>,
    /// (name, sha256 hex).
    pub inputs: Vec<(String, String)>,
    pub upload_every_s: u64,
    pub heartbeat_every_s: u64,
    pub benchmark_seconds: u64,
    pub streams: Vec<Stream>,
    pub raw: Value,
}

impl Manifest2 {
    /// Gives streams with `params.threads = "all"` the machine's thread count.
    pub fn resolve_threads(&mut self, machine: usize) {
        for s in &mut self.streams {
            if s.threads_all {
                s.threads = machine.max(1);
            }
        }
    }
}

/// Chunk `i` of stream `k`: `Rng::derive(run_seed, &[k, i])`, first output.
pub fn chunk_seed(run_seed: u64, k: u64, i: u64) -> u64 {
    Rng::derive(run_seed, &[k, i]).next_u64()
}

/// Manifest format: 1 when `format` is absent.
pub fn format_of(v: &Value) -> Result<u64, String> {
    match v.get("format") {
        None | Some(Value::Null) => Ok(1),
        Some(f) => match f.as_u64() {
            Some(n @ (1 | 2)) => Ok(n),
            _ => Err("`format` must be 1 or 2".into()),
        },
    }
}

pub fn valid_id(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
}

fn f64_of(v: &Value, k: &str, default: f64) -> Result<f64, String> {
    match v.get(k) {
        None => Ok(default),
        Some(x) => x.as_f64().ok_or_else(|| format!("`{k}` must be a number")),
    }
}

/// Parses and checks a format-2 manifest.
pub fn parse_v2(v: &Value) -> Result<Manifest2, String> {
    if format_of(v)? != 2 {
        return Err("not a format-2 manifest".into());
    }
    let run_id = v
        .get("run_id")
        .and_then(Value::as_str)
        .filter(|s| valid_id(s))
        .ok_or("`run_id` is required (letters, digits, - _ .)")?
        .to_string();
    let run_seed = v
        .get("run_seed")
        .and_then(Value::as_u64)
        .ok_or("`run_seed` (whole number) is required")?;
    let deadline = match v.get("deadline_utc") {
        None | Some(Value::Null) => None,
        Some(s) => Some(
            s.as_str()
                .and_then(crate::parse_utc)
                .ok_or("deadline_utc must look like 2026-10-20T06:00:00Z")?,
        ),
    };
    let threads = v
        .get("threads")
        .and_then(Value::as_u64)
        .filter(|&t| t > 0)
        .map(|t| t as usize);
    let mut inputs = Vec::new();
    if let Some(o) = v.get("inputs").and_then(Value::as_object) {
        for (name, h) in o {
            let hex = h
                .as_str()
                .and_then(|s| s.strip_prefix("sha256:"))
                .filter(|x| x.len() == 64 && x.bytes().all(|b| b.is_ascii_hexdigit()))
                .ok_or_else(|| format!("input `{name}` must be \"sha256:<64 hex>\""))?;
            if !valid_id(name) {
                return Err(format!(
                    "input name `{name}` may only use letters, digits, - _ ."
                ));
            }
            inputs.push((name.clone(), hex.to_ascii_lowercase()));
        }
    }
    let upload_every_s = v
        .get("upload_every_s")
        .and_then(Value::as_u64)
        .unwrap_or(600);
    if upload_every_s > 600 {
        return Err("upload_every_s must be at most 600".into());
    }
    let heartbeat_every_s = v
        .get("heartbeat_every_s")
        .and_then(Value::as_u64)
        .unwrap_or(30)
        .max(1);
    let benchmark_seconds = v
        .get("benchmark_seconds")
        .and_then(Value::as_u64)
        .unwrap_or(60);
    let list = v
        .get("streams")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
        .ok_or("`streams` must be a non-empty list")?;
    let mut streams = Vec::new();
    let mut names = BTreeSet::new();
    for s in list {
        let name = s
            .get("name")
            .and_then(Value::as_str)
            .filter(|n| valid_id(n))
            .ok_or("every stream needs a `name` (letters, digits, - _ .)")?
            .to_string();
        if !names.insert(name.clone()) {
            return Err(format!("stream name `{name}` is used twice"));
        }
        // A key the scheduler does not read would be ignored in silence (review E, 2026-10-16:
        // a stream-level `"threads": 32` ran each chunk on one thread), so it is refused.
        if let Some(o) = s.as_object()
            && let Some(k) = o.keys().find(|k| !STREAM_KEYS.contains(&k.as_str()))
        {
            let hint = if PARAMS_KEYS.contains(&k.as_str()) {
                format!(" (it belongs in the stream's params: `params.{k}`)")
            } else {
                String::new()
            };
            return Err(format!(
                "stream `{name}`: unknown key `{k}`{hint}; a stream has only {}",
                STREAM_KEYS.join(", ")
            ));
        }
        let priority = s
            .get("priority")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("stream `{name}` needs a whole-number `priority`"))?;
        let kind = s
            .get("kind")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("stream `{name}` needs a `kind`"))?
            .to_string();
        if !KINDS.contains(&kind.as_str()) {
            return Err(format!("stream `{name}`: unknown kind `{kind}`"));
        }
        let params = s.get("params").cloned().unwrap_or_else(|| json!({}));
        if !params.is_object() {
            return Err(format!("stream `{name}`: `params` must be an object"));
        }
        let chunks = match s.get("chunks") {
            Some(Value::String(x)) if x == "until_decided" => Chunks::UntilDecided,
            Some(x) if x.as_u64().is_some() => Chunks::Fixed(x.as_u64().unwrap_or(0)),
            _ => {
                return Err(format!(
                    "stream `{name}`: `chunks` must be a whole number or \"until_decided\""
                ));
            }
        };
        let (threads, threads_all) = match params.get("threads") {
            None => (1, false),
            Some(Value::String(x)) if x == "all" => (1, true),
            Some(x) => (
                x.as_u64().filter(|&t| t > 0).ok_or_else(|| {
                    format!("stream `{name}`: params.threads must be a whole number or \"all\"")
                })? as usize,
                false,
            ),
        };
        let sizes = match params.get("chunk_sizes") {
            None => None,
            Some(a) => Some(
                a.as_array()
                    .and_then(|a| a.iter().map(Value::as_u64).collect::<Option<Vec<_>>>())
                    .filter(|a| Chunks::Fixed(a.len() as u64) == chunks)
                    .ok_or_else(|| {
                        format!("stream `{name}`: chunk_sizes must list one size per chunk")
                    })?,
            ),
        };
        let est_seconds = f64_of(&params, "est_seconds", 0.0)?;
        let blocks_per_chunk = params
            .get("blocks_per_chunk")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            .max(1);
        let sprt = match params.get("sprt") {
            None => None,
            Some(p) => {
                let sp = SprtParams {
                    s0: f64_of(p, "s0", 0.0)?,
                    s1: f64_of(p, "s1", 0.02)?,
                    alpha: f64_of(p, "alpha", 0.05)?,
                    beta: f64_of(p, "beta", 0.05)?,
                };
                if sp.s1.partial_cmp(&sp.s0) != Some(std::cmp::Ordering::Greater) {
                    return Err(format!("stream `{name}`: sprt s1 must be above s0"));
                }
                let max_blocks = match (p.get("max_blocks").and_then(Value::as_u64), &chunks) {
                    (Some(m), _) => m,
                    (None, Chunks::Fixed(n)) => n * blocks_per_chunk,
                    (None, Chunks::UntilDecided) => {
                        return Err(format!(
                            "stream `{name}`: until_decided needs sprt.max_blocks"
                        ));
                    }
                };
                let reject_only = match p.get("reject_only") {
                    None => false,
                    Some(b) => b.as_bool().ok_or_else(|| {
                        format!("stream `{name}`: sprt.reject_only must be true or false")
                    })?,
                };
                Some(StreamSprt {
                    params: sp,
                    max_blocks,
                    reject_only,
                })
            }
        };
        if chunks == Chunks::UntilDecided && (kind != "arena_blocks" || sprt.is_none()) {
            return Err(format!(
                "stream `{name}`: until_decided needs kind arena_blocks with sprt params"
            ));
        }
        // A hybrid rung's settings are checked here, so a typo stops the run before it starts.
        if kind == "hybrid_rung" {
            crate::hybrid::parse(&params).map_err(|e| format!("stream `{name}`: {e}"))?;
        }
        if kind == "arena_blocks" {
            for k in ["a", "b"] {
                if params.get(k).and_then(Value::as_str).is_none() {
                    return Err(format!("stream `{name}`: arena_blocks needs params.{k}"));
                }
            }
        }
        streams.push(Stream {
            name,
            priority,
            kind,
            params,
            chunks,
            threads,
            threads_all,
            sizes,
            est_seconds,
            blocks_per_chunk,
            sprt,
        });
    }
    Ok(Manifest2 {
        run_id,
        run_seed,
        deadline,
        threads,
        bin_sha: v.get("bin_sha").and_then(Value::as_str).map(str::to_string),
        inputs,
        upload_every_s,
        heartbeat_every_s,
        benchmark_seconds,
        streams,
        raw: v.clone(),
    })
}

// ---------------------------------------------------------------- the sequential test

/// The test over the contiguous finished prefix of an SPRT stream.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefixDecision {
    /// Blocks counted (the prefix, cut at the deciding block).
    pub blocks: u64,
    /// Chunks whose blocks are counted (indices `0..chunks`).
    pub chunks: u64,
    pub llr: f64,
    pub decision: Decision,
    pub mean: f64,
    pub sd: f64,
}

/// Recomputes the test in block order over chunks `0, 1, …` while each is finished; `scores`
/// maps a finished chunk index to its block scores (units of `arena::block::SCORE_DEN`).
/// Finished chunks after a gap, or after the deciding block, never count.
pub fn sprt_prefix(scores: &BTreeMap<u64, Vec<i64>>, sprt: &StreamSprt) -> PrefixDecision {
    let unit = arena::block::SCORE_DEN as f64 * 4_294_967_296.0;
    let mut m = Moments::default();
    let mut out = PrefixDecision {
        blocks: 0,
        chunks: 0,
        llr: 0.0,
        decision: Decision::Continue,
        mean: 0.0,
        sd: 0.0,
    };
    let mut i = 0u64;
    'chunks: while let Some(blocks) = scores.get(&i) {
        out.chunks = i + 1;
        for &x in blocks {
            m.push(x);
            let (mu, var) = m.mean_var(unit);
            out.blocks = m.n;
            out.llr = llr(m.n, mu, var, sprt.params.s0, sprt.params.s1);
            out.mean = mu;
            out.sd = var.sqrt();
            out.decision = decide(
                m.n,
                out.llr,
                &sprt.params,
                sprt.reject_only,
                sprt.max_blocks,
            );
            if out.decision != Decision::Continue {
                break 'chunks;
            }
        }
        i += 1;
    }
    out
}

// ---------------------------------------------------------------- queue

/// What the scheduler knows about one stream.
#[derive(Clone, Debug, Default)]
pub struct StreamState {
    /// Finished chunks (index → seconds).
    pub done: BTreeMap<u64, f64>,
    pub running: BTreeSet<u64>,
    /// Block scores of finished chunks (SPRT streams).
    pub scores: BTreeMap<u64, Vec<i64>>,
}

impl StreamState {
    /// Whether the stream's test has decided (no more chunks are issued).
    pub fn decided(&self, s: &Stream) -> bool {
        s.sprt.as_ref().is_some_and(|sp| {
            s.chunks == Chunks::UntilDecided
                && sprt_prefix(&self.scores, sp).decision != Decision::Continue
        })
    }

    /// The next chunk to issue in this stream, if any.
    pub fn next(&self, s: &Stream) -> Option<u64> {
        if self.decided(s) {
            return None;
        }
        s.order()
            .into_iter()
            .find(|i| !self.done.contains_key(i) && !self.running.contains(i))
    }

    /// Chunks still to issue (an upper bound for until_decided streams).
    pub fn left(&self, s: &Stream) -> u64 {
        if self.decided(s) {
            return 0;
        }
        s.total() - self.done.len() as u64 - self.running.len() as u64
    }
}

/// Median of durations (upper median), or `None`.
pub fn median(v: &[f64]) -> Option<f64> {
    let mut v: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    Some(v[v.len() / 2])
}

/// Running median duration of finished chunks of `s`'s kind (all streams of that kind), if any
/// has finished (on this machine or, through the done-list, an earlier one).
pub fn finished_median(streams: &[Stream], states: &[StreamState], s: &Stream) -> Option<f64> {
    let all: Vec<f64> = streams
        .iter()
        .zip(states)
        .filter(|(t, _)| t.kind == s.kind)
        .flat_map(|(_, st)| st.done.values().copied())
        .collect();
    median(&all)
}

/// The duration the deadline rule assumes for the next chunk of `s`: the running median of its
/// kind, else the stream's estimate (`params.est_seconds`, default 0).
pub fn kind_median(streams: &[Stream], states: &[StreamState], s: &Stream) -> f64 {
    finished_median(streams, states, s).unwrap_or(s.est_seconds)
}

/// The chunk-size target (M3-SPEC §7.2: 2–5 minutes): a note for the heartbeat and the log when
/// a kind's running median is outside it, so the next manifest can resize its chunks. Chunk sizes
/// are never changed inside a run, because a chunk's output must not depend on the machine.
pub fn size_note(median_s: f64) -> Option<String> {
    let min = median_s / 60.0;
    if min.is_nan() || min <= 0.0 {
        None
    } else if min < 2.0 {
        Some(format!(
            "chunks take {min:.1} min, under the 2-5 min target: use larger chunks next time"
        ))
    } else if min > 5.0 {
        Some(format!(
            "chunks take {min:.1} min, over the 2-5 min target: use smaller chunks next time"
        ))
    } else {
        None
    }
}

/// The next chunk to start: the lowest priority number (then stream order) with a chunk left
/// whose threads fit in `free` and which can finish before the deadline
/// (`now + 1.5 × median duration ≤ deadline`).
pub fn pick(
    streams: &[Stream],
    states: &[StreamState],
    free: usize,
    now: u64,
    deadline: Option<u64>,
) -> Option<(usize, u64)> {
    let mut order: Vec<usize> = (0..streams.len()).collect();
    order.sort_by_key(|&k| (streams[k].priority, k));
    for k in order {
        let s = &streams[k];
        if s.threads > free {
            continue;
        }
        let Some(i) = states[k].next(s) else {
            continue;
        };
        if let Some(d) = deadline {
            let need = 1.5 * kind_median(streams, states, s);
            if now as f64 + need > d as f64 {
                continue;
            }
        }
        return Some((k, i));
    }
    None
}

/// The highest-sequence done-list among `keys` (`done/<run>/<seq:08>-<utc>.json`).
pub fn latest_done(keys: &[String]) -> Option<(u64, String)> {
    keys.iter()
        .filter_map(|k| {
            let name = k.rsplit('/').next()?;
            let stem = name.strip_suffix(".json")?;
            let (seq, _) = stem.split_once('-')?;
            (seq.len() == 8)
                .then(|| seq.parse::<u64>().ok())
                .flatten()
                .map(|n| (n, k.clone()))
        })
        .max_by_key(|(n, _)| *n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m2(streams: Value) -> Value {
        json!({"format": 2, "run_id": "t", "run_seed": 7, "streams": streams})
    }

    fn arena_stream(chunks: Value, max_blocks: u64) -> Value {
        json!({"name": "gate", "priority": 2, "kind": "arena_blocks", "chunks": chunks,
               "params": {"a": "rules-v1", "b": "random", "blocks_per_chunk": 2,
                          "sprt": {"s0": 0.0, "s1": 0.02, "max_blocks": max_blocks}}})
    }

    #[test]
    fn parses_format_1_and_2() {
        let v1 = json!({"run_id": "x", "chunk": {"difftest": {"matches": 1, "seed": 1}}});
        assert_eq!(format_of(&v1).unwrap(), 1);
        assert!(parse_v2(&v1).is_err());
        let spec = json!({ "format": 2, "run_id": "m3-teacher-smoke", "run_seed": 7,
            "deadline_utc": "2026-10-20T06:00:00Z", "threads": null, "bin_sha": "abc",
            "inputs": { "equity": format!("sha256:{}", "a".repeat(64)) },
            "upload_every_s": 600, "heartbeat_every_s": 30,
            "streams": [
              { "name": "teacher", "priority": 1, "kind": "teacher", "params": { "hands_per_chunk": 200, "config": "pimc-teacher" }, "chunks": 50 },
              { "name": "gate", "priority": 2, "kind": "arena_blocks", "params": { "a": "pimc-v0", "b": "rules-v1", "block14": true, "blocks_per_chunk": 4, "sprt": { "s0": 0.02, "s1": 0.06, "alpha": 0.05, "beta": 0.05, "max_blocks": 578 } }, "chunks": "until_decided" },
              { "name": "suitability", "priority": 3, "kind": "suitability", "params": { "hands_per_chunk": 250 }, "chunks": 40 } ] });
        let m = parse_v2(&spec).unwrap();
        assert_eq!(m.streams.len(), 3);
        assert_eq!(m.threads, None);
        assert_eq!(m.deadline, crate::parse_utc("2026-10-20T06:00:00Z"));
        assert_eq!(m.inputs[0].0, "equity");
        assert_eq!(m.streams[1].chunks, Chunks::UntilDecided);
        assert_eq!(m.streams[1].total(), 145); // ceil(578 / 4)
        assert_eq!(m.streams[1].blocks_in(144), 2);
        assert_eq!(m.streams[0].total(), 50);
        // Bad inputs.
        assert!(
            parse_v2(&m2(
                json!([{"name": "a", "priority": 1, "kind": "nope", "chunks": 1}])
            ))
            .is_err()
        );
        assert!(
            parse_v2(&m2(
                json!([{"name": "a", "priority": 1, "kind": "teacher", "chunks": "until_decided"}])
            ))
            .is_err()
        );
        assert!(
            parse_v2(&m2(json!([
            {"name": "a", "priority": 1, "kind": "teacher", "chunks": 1},
            {"name": "a", "priority": 2, "kind": "teacher", "chunks": 1}])))
            .is_err()
        );
        assert!(parse_v2(&json!({"format": 3})).is_err());
        // A stream-level `threads` (review E: the table measurement's manifest) is refused, not
        // silently run on one thread per chunk.
        let e = parse_v2(&m2(
            json!([{"name": "tt12g", "priority": 1, "kind": "search_measure",
                                     "threads": 32, "chunks": 1, "params": {"tt_mb": 12288}}]),
        ))
        .expect_err("stream-level threads");
        assert!(
            e.contains("unknown key `threads`") && e.contains("params.threads"),
            "{e}"
        );
        assert!(
            parse_v2(&m2(json!([{"name": "a", "priority": 1, "kind": "difftest",
                                     "chunks": 1, "note": "x"}])))
            .is_err()
        );
        let ok = parse_v2(&m2(
            json!([{"name": "tt12g", "priority": 1, "kind": "search_measure",
                                      "chunks": 1, "params": {"threads": 32, "tt_mb": 12288}}]),
        ))
        .expect("params.threads");
        assert_eq!(ok.streams[0].threads, 32);
    }

    fn streams(v: Value) -> Vec<Stream> {
        parse_v2(&m2(v)).unwrap().streams
    }

    #[test]
    fn priority_order() {
        let s = streams(json!([
            {"name": "low", "priority": 3, "kind": "suitability", "chunks": 2},
            {"name": "high", "priority": 1, "kind": "teacher", "chunks": 2},
            {"name": "mid", "priority": 2, "kind": "agree", "chunks": 1}]));
        let mut st = vec![StreamState::default(); 3];
        let mut got = vec![];
        while let Some((k, i)) = pick(&s, &st, 1, 0, None) {
            st[k].running.insert(i);
            got.push((s[k].name.as_str(), i));
        }
        assert_eq!(
            got,
            [("high", 0), ("high", 1), ("mid", 0), ("low", 0), ("low", 1)]
        );
        // A finished high-priority chunk does not reappear; threads that do not fit are skipped.
        let s2 = streams(json!([
            {"name": "big", "priority": 1, "kind": "teacher", "chunks": 1, "params": {"threads": 4}},
            {"name": "small", "priority": 2, "kind": "agree", "chunks": 1}]));
        let st2 = vec![StreamState::default(); 2];
        assert_eq!(pick(&s2, &st2, 2, 0, None), Some((1, 0)));
        assert_eq!(pick(&s2, &st2, 4, 0, None), Some((0, 0)));
    }

    #[test]
    fn lpt_order() {
        let s = streams(
            json!([{"name": "fixed", "priority": 1, "kind": "teacher", "chunks": 5,
            "params": {"chunk_sizes": [3, 9, 1, 9, 4]}}]),
        );
        assert_eq!(s[0].order(), vec![1, 3, 4, 0, 2]);
        let mut st = vec![StreamState::default()];
        let mut got = vec![];
        while let Some((_, i)) = pick(&s, &st, 1, 0, None) {
            st[0].running.insert(i);
            got.push(i);
        }
        assert_eq!(got, vec![1, 3, 4, 0, 2]);
        // Bad sizes list.
        assert!(
            parse_v2(&m2(
                json!([{"name": "f", "priority": 1, "kind": "teacher", "chunks": 2,
            "params": {"chunk_sizes": [1]}}])
            ))
            .is_err()
        );
    }

    #[test]
    fn deadline_cutoff_with_fake_clock() {
        let s = streams(json!([
            {"name": "slow", "priority": 1, "kind": "teacher", "chunks": 10, "params": {"est_seconds": 100.0}},
            {"name": "fast", "priority": 2, "kind": "agree", "chunks": 10}]));
        let mut st = vec![StreamState::default(); 2];
        let deadline = Some(1000);
        // Estimate 100 s: 1.5 × 100 = 150, so issue up to now = 850.
        assert_eq!(pick(&s, &st, 1, 850, deadline), Some((0, 0)));
        assert_eq!(pick(&s, &st, 1, 851, deadline), Some((1, 0)));
        // Finished chunks set the running median: 200, 300, 400 → 300 → 450 needed.
        for (i, d) in [(0u64, 200.0), (1, 300.0), (2, 400.0)] {
            st[0].done.insert(i, d);
        }
        assert_eq!(pick(&s, &st, 1, 550, deadline), Some((0, 3)));
        assert_eq!(pick(&s, &st, 1, 551, deadline), Some((1, 0)));
        // Past the deadline nothing (the fast stream has median 0 until a chunk finishes).
        st[1].done.insert(0, 10.0);
        assert_eq!(pick(&s, &st, 1, 1000, deadline), None);
        assert_eq!(pick(&s, &st, 1, 985, deadline), Some((1, 1)));
    }

    #[test]
    fn sprt_stops_at_bound_and_ignores_beyond_prefix() {
        let s = streams(json!([arena_stream(json!("until_decided"), 1000)]));
        let sp = s[0].sprt.clone().unwrap();
        let unit = arena::block::SCORE_DEN * 4_294_967_296;
        // A clearly winning candidate: block scores alternate 0.04 and 0.02 (mean 0.03 > s1).
        let good = |j: u64| {
            if j.is_multiple_of(2) {
                unit * 4 / 100
            } else {
                unit * 2 / 100
            }
        };
        let mut st = StreamState::default();
        // Chunks finish out of order: 0, 2, 3 … but 1 is missing, so only chunk 0 counts.
        for i in [0u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] {
            st.scores.insert(i, vec![good(2 * i), good(2 * i + 1)]);
            st.done.insert(i, 1.0);
        }
        let p = sprt_prefix(&st.scores, &sp);
        assert_eq!((p.blocks, p.chunks, p.decision), (2, 1, Decision::Continue));
        assert!(!st.decided(&s[0]));
        assert_eq!(st.next(&s[0]), Some(1));
        // Chunk 1 arrives: the prefix grows until the bound (MIN_BLOCKS = 20 blocks first).
        st.scores.insert(1, vec![good(2), good(3)]);
        st.done.insert(1, 1.0);
        let p = sprt_prefix(&st.scores, &sp);
        assert_eq!(p.decision, Decision::Accept);
        assert_eq!(p.blocks, arena::sprt::MIN_BLOCKS);
        assert_eq!(p.chunks, 10);
        assert!(st.decided(&s[0]));
        assert_eq!(st.next(&s[0]), None);
        assert_eq!(st.left(&s[0]), 0);
        // Same decision whatever the completion order: build it in another order.
        let mut other = BTreeMap::new();
        for i in (0u64..13).rev() {
            other.insert(i, st.scores[&i].clone());
        }
        assert_eq!(sprt_prefix(&other, &sp), p);
        // Losing candidate rejects.
        let mut bad = BTreeMap::new();
        for i in 0u64..15 {
            bad.insert(i, vec![-unit * 2 / 100, -unit * 4 / 100]);
        }
        assert_eq!(sprt_prefix(&bad, &sp).decision, Decision::Reject);
        // The cap.
        let cap = streams(json!([arena_stream(json!("until_decided"), 3)]));
        let cs = cap[0].sprt.clone().unwrap();
        assert_eq!(cap[0].total(), 2);
        assert_eq!(cap[0].blocks_in(1), 1);
        let mut z = BTreeMap::new();
        z.insert(0u64, vec![0i64, 0]);
        assert_eq!(sprt_prefix(&z, &cs).decision, Decision::Continue);
        z.insert(1, vec![0]);
        assert_eq!(sprt_prefix(&z, &cs).decision, Decision::Inconclusive);
    }

    #[test]
    fn done_list_versioning_picks_highest_sequence() {
        let keys: Vec<String> = [
            "done/r/00000002-20261014T010000Z.json",
            "done/r/00000010-20261014T000000Z.json",
            "done/r/00000009-20261015T000000Z.json",
            "done/r/junk.json",
            "done/r/123-x.json",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(
            latest_done(&keys),
            Some((10, "done/r/00000010-20261014T000000Z.json".into()))
        );
        assert_eq!(latest_done(&[]), None);
    }

    #[test]
    fn threads_all_and_reject_only() {
        let mut m = parse_v2(&m2(json!([
            {"name": "box", "priority": 1, "kind": "solver_spike", "chunks": 1, "params": {"threads": "all"}},
            {"name": "one", "priority": 2, "kind": "difftest", "chunks": 1, "params": {"threads": 2}}])))
        .unwrap();
        assert!(m.streams[0].threads_all);
        m.resolve_threads(16);
        assert_eq!((m.streams[0].threads, m.streams[1].threads), (16, 2));
        assert!(
            parse_v2(&m2(
                json!([{"name": "a", "priority": 1, "kind": "teacher", "chunks": 1, "params": {"threads": 0}}])
            ))
            .is_err()
        );
        // Stage 1 (reject-only): a clear winner never accepts, it runs to the cap.
        let mut v = arena_stream(json!("until_decided"), 40);
        v["params"]["sprt"]["reject_only"] = json!(true);
        let s = streams(json!([v]));
        let sp = s[0].sprt.clone().unwrap();
        assert!(sp.reject_only);
        let unit = arena::block::SCORE_DEN * 4_294_967_296;
        let mut good = BTreeMap::new();
        let mut bad = BTreeMap::new();
        for i in 0u64..20 {
            good.insert(i, vec![unit * 4 / 100, unit * 2 / 100]);
            bad.insert(i, vec![-unit * 2 / 100, -unit * 4 / 100]);
        }
        let p = sprt_prefix(&good, &sp);
        assert_eq!((p.decision, p.blocks), (Decision::Inconclusive, 40));
        assert_eq!(sprt_prefix(&bad, &sp).decision, Decision::Reject);
    }

    #[test]
    fn size_target_notes() {
        assert_eq!(size_note(0.0), None);
        assert_eq!(size_note(180.0), None);
        assert!(size_note(30.0).unwrap().contains("larger"));
        assert!(size_note(600.0).unwrap().contains("smaller"));
    }

    #[test]
    fn chunk_seeds_are_labelled() {
        assert_eq!(chunk_seed(7, 1, 2), Rng::derive(7, &[1, 2]).next_u64());
        assert_ne!(chunk_seed(7, 1, 2), chunk_seed(7, 2, 1));
    }
}
