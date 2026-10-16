//! A quantised oracle network as a solver frontier evaluator: canonical position → encoder →
//! integer MLP → money chance (rotated back to the real teams). Two heads (metadata `head`):
//! - `money4` (spec §6; the default when absent): four logits, an integer softmax;
//! - `outcome-v1` (spec §6a): one logit per outcome of the position's trick
//!   ([`crate::outcome`]), an integer softmax over the possible ones, then the mixture of their
//!   exact money vectors.
//!
//! Weights file (`41bots-int-mlp-v1`, safetensors): tensors `w0, b0, w1, b1, …` (`I32`
//! `[n_in, n_out]` input-major, `I64` `[n_out]`), and metadata `format`, `encoder` (must be
//! [`ENCODER`]), `n_features`, `shifts` (a JSON list, one right shift per layer), `trick` (the
//! frontier trick the network was trained for), plus whatever training recorded (data hashes,
//! sizes). The file's SHA-256 names the network in hybrid values (spec §6: the values version
//! names the net).
//!
//! **Evaluation** ([`Oracle::eval`]) gives exactly the plain path ([`Oracle::eval_reference`]:
//! encode, the reference forward pass, the outcome table, the integer softmax), faster:
//! - the fast forward pass ([`crate::fast`]), with this thread's incremental first layer;
//! - everything that depends only on the position's [`DenseKey`] (the 100 equity inputs and the
//!   first layer's dense sums) and on the key with the top trump's team (the outcome table) is
//!   computed once and kept, per thread and oracle (a few dozen keys occur in a solve);
//! - a memo of recent positions (direct-mapped, `2^14` slots per thread and oracle, keyed by
//!   the full canonical form and match context): the solver's transposition table already
//!   values a repeated *position* once, but different positions with the same canonical form
//!   (10–40% of the frontier requests of the solves we measured at tricks 2–4) were each evaluated.
//!
//! Each cache holds a pure function of its key and compares the whole key, so what it holds
//! (which depends on thread scheduling and eviction) never changes a value.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use equity::MoneyChance;
use equity::sha256::sha256_hex;
use solver::frontier::{CanonPos, Frontier};
use solver::{Leaf, MatchCtx};

use crate::encode::{
    DenseKey, ENCODER, N_FEATURES, binary_features, dense_features, encode, rotation, unrotate,
};
use crate::fast::{FastMlp, Scratch};
use crate::mlp::{IntMlp, Layer};
use crate::money::money_from_logits;
use crate::outcome::{
    Outcomes, mix_outcomes, money_from_outcomes, outcomes, outcomes_of, top_team, vocab,
};
use crate::safetensors;

/// The network's output head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Head {
    /// Four money logits (spec §6).
    Money4,
    /// One logit per outcome (spec §6a).
    Outcome,
}

impl Head {
    /// Its name in weights files.
    pub fn name(self) -> &'static str {
        match self {
            Head::Money4 => "money4",
            Head::Outcome => "outcome-v1",
        }
    }

    /// The head of a weights file (`money4` when the file names none).
    pub fn of(f: &safetensors::File) -> Result<Head, String> {
        match f.meta.get("head").map(String::as_str) {
            None | Some("money4") => Ok(Head::Money4),
            Some("outcome-v1") => Ok(Head::Outcome),
            Some(h) => Err(format!("unknown head {h}")),
        }
    }
}

/// The weights format this crate reads.
pub const FORMAT: &str = "41bots-int-mlp-v1";

/// A loaded oracle network.
pub struct Oracle {
    net: IntMlp,
    fast: FastMlp,
    leaf: Arc<dyn Leaf + Send + Sync>,
    /// Identifies this oracle's per-thread caches.
    id: u64,
    /// Whether [`Oracle::eval`] remembers recent positions' values.
    memo: bool,
    /// The output head.
    pub head: Head,
    /// The frontier trick the network was trained for.
    pub trick: u8,
    /// SHA-256 of the weights file (hex).
    pub sha256: String,
    /// The file's metadata.
    pub meta: BTreeMap<String, String>,
}

/// Parse the integer network of a weights file.
pub fn parse_net(f: &safetensors::File) -> Result<IntMlp, String> {
    if f.meta("format")? != FORMAT {
        return Err(format!("format {} is not {FORMAT}", f.meta("format")?));
    }
    if f.meta("encoder")? != ENCODER {
        return Err(format!(
            "network made for encoder {}; this build encodes {ENCODER}",
            f.meta("encoder")?
        ));
    }
    if f.meta("n_features")? != N_FEATURES.to_string() {
        return Err(format!("network takes {} features", f.meta("n_features")?));
    }
    let shifts: Vec<u32> =
        serde_json::from_str(f.meta("shifts")?).map_err(|e| format!("shifts: {e}"))?;
    let mut layers = Vec::new();
    for (k, &shift) in shifts.iter().enumerate() {
        let w = f.get(&format!("w{k}"))?;
        let b = f.get(&format!("b{k}"))?;
        if w.shape.len() != 2 || b.shape.len() != 1 || b.shape[0] != w.shape[1] {
            return Err(format!("layer {k}: shapes {:?} {:?}", w.shape, b.shape));
        }
        layers.push(Layer {
            n_in: w.shape[0],
            n_out: w.shape[1],
            w: w.as_i32()?,
            b: b.as_i64()?,
            shift,
        });
    }
    if f.tensors.len() != 2 * layers.len() {
        return Err("tensors other than the layers".into());
    }
    IntMlp::new(layers)
}

/// A weights file for an integer network (the format [`parse_net`] reads), with the frontier
/// trick and any extra metadata (a `head` entry names the head; none means `money4`).
pub fn write_net(net: &IntMlp, trick: u8, extra: &BTreeMap<String, String>) -> Vec<u8> {
    let mut f = safetensors::File::default();
    for (k, l) in net.layers().iter().enumerate() {
        f.tensors.insert(
            format!("w{k}"),
            safetensors::Tensor::i32(vec![l.n_in, l.n_out], &l.w),
        );
        f.tensors.insert(
            format!("b{k}"),
            safetensors::Tensor::i64(vec![l.n_out], &l.b),
        );
    }
    f.meta = extra.clone();
    f.meta.insert("format".into(), FORMAT.into());
    f.meta.insert("encoder".into(), ENCODER.into());
    f.meta.insert("n_features".into(), N_FEATURES.to_string());
    f.meta.insert("trick".into(), trick.to_string());
    let shifts: Vec<u32> = net.layers().iter().map(|l| l.shift).collect();
    f.meta.insert(
        "shifts".into(),
        serde_json::to_string(&shifts).expect("a list of numbers"),
    );
    safetensors::write(&f)
}

impl Oracle {
    /// Load a weights file; `leaf` (the equity table) gives the encoder's dense features.
    pub fn load(path: &Path, leaf: Arc<dyn Leaf + Send + Sync>) -> Result<Oracle, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Oracle::from_bytes(&bytes, leaf).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// [`Oracle::load`] from the file's bytes.
    pub fn from_bytes(bytes: &[u8], leaf: Arc<dyn Leaf + Send + Sync>) -> Result<Oracle, String> {
        let f = safetensors::read(bytes)?;
        let net = parse_net(&f)?;
        let trick: u8 = f
            .meta("trick")?
            .parse()
            .map_err(|_| "trick: not a number".to_string())?;
        let head = Head::of(&f)?;
        let want = match head {
            Head::Money4 => 4,
            Head::Outcome => vocab(trick),
        };
        if net.n_out() != want {
            return Err(format!(
                "a {} head for trick {trick} has {want} outputs, not {}",
                head.name(),
                net.n_out()
            ));
        }
        Ok(Oracle {
            fast: FastMlp::new(net.clone()),
            net,
            leaf,
            id: NEXT_ORACLE.fetch_add(1, Ordering::Relaxed),
            memo: true,
            head,
            trick,
            sha256: sha256_hex(bytes),
            meta: f.meta,
        })
    }

    /// The integer network.
    pub fn net(&self) -> &IntMlp {
        &self.net
    }

    /// The fast network (the forward pass [`Oracle::eval`] uses).
    pub fn fast(&self) -> &FastMlp {
        &self.fast
    }

    /// Turn the memo of recent positions on or off (on by default; benchmarks turn it off to
    /// time the network itself). Values never depend on it.
    pub fn set_memo(&mut self, on: bool) {
        self.memo = on;
    }

    /// The network's money chance of a canonical position (real team order): exactly
    /// [`Oracle::eval_reference`], computed faster (see the module notes of [`crate::fast`] and
    /// [`EvalStats`]): the fast forward pass with an incremental first layer, the first layer's
    /// dense part and the outcome table computed once per [`DenseKey`], and a memo of recent
    /// positions. The caches are per thread and per oracle; their contents never change a value
    /// (each holds a pure function of its key, the key compared in full).
    ///
    /// # Panics
    /// With the outcome head, for a position of another trick than the network's.
    pub fn eval(&self, c: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        if self.head == Head::Outcome {
            assert_eq!(
                c.tricks_done + 1,
                self.trick,
                "a position of the net's trick"
            );
        }
        STATES.with(|cell| {
            let mut states = cell.borrow_mut();
            let i = match states.iter().position(|s| s.oracle == self.id) {
                Some(i) => i,
                None => {
                    if states.len() >= MAX_STATES {
                        states.remove(0);
                    }
                    states.push(EvalState::new(self.id));
                    states.len() - 1
                }
            };
            states[i].eval(self, c, ctx)
        })
    }

    /// This thread's cache counters for this oracle (reports only).
    pub fn thread_stats(&self) -> EvalStats {
        STATES.with(|cell| {
            cell.borrow()
                .iter()
                .find(|s| s.oracle == self.id)
                .map_or(EvalStats::default(), |s| s.stats)
        })
    }

    /// The network's money chance computed the plain way: [`encode`], [`IntMlp::forward`] (the
    /// reference forward pass), then [`outcomes`] and [`money_from_outcomes`] (or
    /// [`money_from_logits`]). [`Oracle::eval`] gives exactly this, faster.
    ///
    /// # Panics
    /// With the outcome head, for a position of another trick than the network's.
    pub fn eval_reference(&self, c: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        let (f, r) = encode(c, ctx, &*self.leaf);
        let z = self.net.forward(&f);
        let m = match self.head {
            Head::Money4 => money_from_logits(&z),
            Head::Outcome => {
                assert_eq!(
                    c.tricks_done + 1,
                    self.trick,
                    "a position of the net's trick"
                );
                money_from_outcomes(&z, &outcomes(c, ctx, &*self.leaf))
            }
        };
        unrotate(m, r)
    }
}

static NEXT_ORACLE: AtomicU64 = AtomicU64::new(1);

/// Oracles whose caches a thread keeps at once.
const MAX_STATES: usize = 4;
/// Slots of the memo of recent positions (direct-mapped).
const MEMO_SLOTS: usize = 1 << 14;
/// Dense-part vectors kept per thread and oracle (the cache is emptied when full).
const DENSE_CAP: usize = 256;
/// Outcome tables kept per thread and oracle (the cache is emptied when full).
const OUTCOME_CAP: usize = 4096;

thread_local! {
    static STATES: RefCell<Vec<EvalState>> = const { RefCell::new(Vec::new()) };
}

/// Counters of one thread's evaluations by one oracle (reports only).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EvalStats {
    /// Positions asked for.
    pub requests: u64,
    /// Answered by the memo of recent positions.
    pub memo_hits: u64,
    /// Network forward passes.
    pub forwards: u64,
    /// Dense parts computed (encoder equity features and the first layer's dense sums).
    pub dense_misses: u64,
    /// Outcome tables computed.
    pub outcome_misses: u64,
}

/// FxHash's multiply-rotate step: a fast hash for the caches' lookups (never iterated, so the
/// hash cannot change a value).
#[derive(Default)]
struct FxHasher(u64);

impl FxHasher {
    #[inline]
    fn add(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl Hasher for FxHasher {
    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for w in words {
            self.add(u64::from_le_bytes(*w));
        }
        if !rest.is_empty() {
            let mut b = [0u8; 8];
            b[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(b));
        }
    }
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }
    fn write_u16(&mut self, i: u16) {
        self.add(u64::from(i));
    }
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

type Fx = BuildHasherDefault<FxHasher>;

/// A memo entry: the full key and the value.
#[derive(Clone, Copy)]
struct MemoEntry {
    c: CanonPos,
    games: [u8; 4],
    dealer: u8,
    v: MoneyChance,
}

/// One thread's caches for one oracle.
struct EvalState {
    oracle: u64,
    memo: Vec<Option<MemoEntry>>,
    dense: HashMap<DenseKey, Box<[i64]>, Fx>,
    outs: HashMap<(DenseKey, Option<u8>), Outcomes, Fx>,
    scratch: Scratch,
    stats: EvalStats,
}

impl EvalState {
    fn new(oracle: u64) -> Self {
        EvalState {
            oracle,
            memo: Vec::new(),
            dense: HashMap::default(),
            outs: HashMap::default(),
            scratch: Scratch::new(),
            stats: EvalStats::default(),
        }
    }

    fn eval(&mut self, o: &Oracle, c: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        self.stats.requests += 1;
        let slot = if o.memo {
            if self.memo.is_empty() {
                self.memo = vec![None; MEMO_SLOTS];
            }
            let mut h = FxHasher::default();
            std::hash::Hash::hash(c, &mut h);
            h.write(&ctx.games);
            h.write_u8(ctx.dealer_team);
            let slot = (h.finish() >> 32) as usize & (MEMO_SLOTS - 1);
            if let Some(e) = &self.memo[slot]
                && e.c == *c
                && e.games == ctx.games
                && e.dealer == ctx.dealer_team
            {
                self.stats.memo_hits += 1;
                return e.v;
            }
            Some(slot)
        } else {
            None
        };
        let r = rotation(c);
        let key = DenseKey::of(c, ctx);
        let f = binary_features(c, ctx);
        if self.dense.len() >= DENSE_CAP && !self.dense.contains_key(&key) {
            self.dense.clear();
        }
        let base = self.dense.entry(key).or_insert_with(|| {
            self.stats.dense_misses += 1;
            let d = dense_features(&key, &*o.leaf);
            let mut base = vec![0i64; o.fast.base_len()];
            o.fast.base(&d, &mut self.scratch, &mut base);
            base.into_boxed_slice()
        });
        let z = o.fast.forward_base(f.active(), base, &mut self.scratch);
        self.stats.forwards += 1;
        let m = match o.head {
            Head::Money4 => money_from_logits(&z),
            Head::Outcome => {
                let top = top_team(c);
                if self.outs.len() >= OUTCOME_CAP && !self.outs.contains_key(&(key, top)) {
                    self.outs.clear();
                }
                let t = self.outs.entry((key, top)).or_insert_with(|| {
                    self.stats.outcome_misses += 1;
                    outcomes_of(&key, top, &*o.leaf)
                });
                mix_outcomes(&z, t.n, t.mask, &t.vecs)
            }
        };
        let v = unrotate(m, r);
        if let Some(slot) = slot {
            self.memo[slot] = Some(MemoEntry {
                c: *c,
                games: ctx.games,
                dealer: ctx.dealer_team,
                v,
            });
        }
        v
    }
}

impl Frontier for Oracle {
    fn value(&self, pos: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        self.eval(pos, ctx)
    }
}
