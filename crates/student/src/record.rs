//! The teacher's stored records (docs/STUDENT.md §3, format `41bots-student-teacher-v1`).
//!
//! A shard is two `.npy` files of fixed little-endian fields (PLAN §12.7) and a JSON sidecar:
//! - `NAME.npy`: one record per decision ([`RECORD_DESCR`]): the position in raw form
//!   ([`RawPosition`]), its identity (`pos_sha256`), source, generator seed and index, trick,
//!   place in the trick, stratum, decision kind, the candidates ([`cand_code`]), the deals (the
//!   decision seed, the first deal index, the count, the tracker's world count), the per-candidate
//!   means and own-pair SDs over the record's deals, and where its values are in the values file
//!   with their SHA-256;
//! - `NAME.values.npy` (`u64 [rows, 4]`): for each record, deal by deal, candidate by candidate,
//!   the four pairs' money chance in real pair order (units of 2^-32);
//! - `NAME.npy.json`: the stamps (teacher, equity table, rules and values versions, binary,
//!   sampler, deals, generator settings) and both files' SHA-256.
//!
//! The shard's key is the SHA-256 of `NAME.npy`. Every record holds the SHA-256 of its own value
//! rows, so that one key covers the values too; [`ShardReader::values`] checks them on every read.
//! Means are recomputed from the values (`MoneyChance::mix`, deal order) and checked against the
//! stored ones; nothing a reader returns is taken on trust.

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use engine::cards::DECK_SIZE;
use engine::game::HandStartState;
use equity::MoneyChance;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::position::{
    Cand, MAX_ACTIONS, RawPosition, action_code, action_from_code, cand_code, cand_from_code,
};

/// The format name, in every sidecar.
pub const FORMAT: &str = "41bots-student-teacher-v1";

/// Most candidates a decision has (a robber's joint (burn, play) decisions: at most 5 burns × 5
/// plays).
pub const MAX_CANDS: usize = 32;

/// Decision kinds (the codes of `search::teacher`): a play.
pub const KIND_PLAY: u8 = 0;
/// A robber's joint (burn, first card) decision.
pub const KIND_JOINT: u8 = 2;

/// Where positions come from: `rules-v1` self-play (the yardstick's generator).
pub const SOURCE_RULES: u8 = 1;
/// `rules-v1` self-play with each action before the decision random with probability 1/4.
pub const SOURCE_MIXED: u8 = 2;
/// Random legal play before the decision.
pub const SOURCE_RANDOM: u8 = 3;
/// The student's own games (DAgger rounds).
pub const SOURCE_STUDENT: u8 = 4;
/// The app's recorded hands.
pub const SOURCE_APP: u8 = 5;

/// A source's name.
pub fn source_name(s: u8) -> &'static str {
    match s {
        SOURCE_RULES => "rules",
        SOURCE_MIXED => "mixed",
        SOURCE_RANDOM => "random",
        SOURCE_STUDENT => "student",
        SOURCE_APP => "app",
        _ => "unknown",
    }
}

/// A source by name.
pub fn parse_source(s: &str) -> Result<u8, String> {
    match s {
        "rules" => Ok(SOURCE_RULES),
        "mixed" => Ok(SOURCE_MIXED),
        "random" => Ok(SOURCE_RANDOM),
        "student" => Ok(SOURCE_STUDENT),
        "app" => Ok(SOURCE_APP),
        _ => Err(format!(
            "unknown source {s} (rules, mixed, random, student or app)"
        )),
    }
}

/// The record type of `NAME.npy`, as NumPy writes it.
pub const RECORD_DESCR: &str = "[('pos_sha256', '|u1', (32,)), ('source', '|u1'), ('trick', '|u1'), \
('place', '|u1'), ('seat', '|u1'), ('stratum', '|u1'), ('kind', '|u1'), ('n_cands', '|u1'), \
('n_actions', '|u1'), ('gen_seed', '<u8'), ('index', '<u8'), ('dealer', '|u1'), \
('scores', '|u1', (4,)), ('bonus_taken', '|u1', (4,)), ('games_won', '|u1', (4,)), \
('game_index', '|u1'), ('hand_index', '<u4'), ('deck', '|u1', (53,)), ('actions', '<u2', (48,)), \
('cands', '<u2', (32,)), ('dseed', '<u8'), ('deal_from', '<u4'), ('n_deals', '<u4'), \
('log2_worlds_q8', '<u2'), ('means', '<u8', (32, 4)), ('own_sd', '<u8', (32,)), \
('values_at', '<u8'), ('values_sha256', '|u1', (32,))]";

/// Bytes per record.
pub const RECORD_BYTES: usize = 32
    + 8
    + 16
    + (1 + 4 + 4 + 4 + 1 + 4)
    + DECK_SIZE
    + 2 * MAX_ACTIONS
    + 2 * MAX_CANDS
    + 16
    + 2
    + 32 * MAX_CANDS
    + 8 * MAX_CANDS
    + 8
    + 32;

/// One decision as the teacher valued it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TeacherRecord {
    /// The position.
    pub pos: RawPosition,
    /// Where it came from (`SOURCE_*`).
    pub source: u8,
    /// The generator's seed.
    pub gen_seed: u64,
    /// The decision's index in its generator's stream.
    pub index: u64,
    /// The trick (1..=5).
    pub trick: u8,
    /// Cards already in the trick (0 = the leader).
    pub place: u8,
    /// The hand start's match-state stratum (1..=4, `search::measure::stratum_of`).
    pub stratum: u8,
    /// [`KIND_PLAY`] or [`KIND_JOINT`].
    pub kind: u8,
    /// The candidates, in the engine's candidate order.
    pub cands: Vec<Cand>,
    /// The decision seed: deal `w` is drawn from `Rng::derive(dseed, [WORLD, w])`.
    pub dseed: u64,
    /// The first deal's index.
    pub deal_from: u32,
    /// `⌊256·log2 N⌋` of the card tracker's exact count of consistent deals (0 if not counted).
    pub log2_worlds_q8: u16,
    /// `values[j][c]`: deal `deal_from + j`, candidate `c`, in real pair order.
    pub values: Vec<Vec<MoneyChance>>,
}

/// The deciding seat's pair.
fn team(seat: u8) -> usize {
    usize::from(engine::team_of(seat))
}

impl TeacherRecord {
    /// Deals valued.
    pub fn n_deals(&self) -> u32 {
        self.values.len() as u32
    }

    /// Per candidate, the plain mean over the deals (`MoneyChance::mix`, weight 1, deal order:
    /// exactly `search::pimc::aggregate`'s).
    pub fn means(&self) -> Vec<MoneyChance> {
        means_of(&self.values, self.cands.len())
    }

    /// Per candidate, the sample SD over the deals of the deciding pair's value, in units of
    /// 2^-32 (integer square root, as `search::pimc::aggregate`).
    pub fn own_sd(&self) -> Vec<u64> {
        let t = team(self.pos.seat);
        (0..self.cands.len())
            .map(|c| {
                let n = self.values.len() as u128;
                let (s, s2) = self.values.iter().fold((0u128, 0u128), |(a, b), r| {
                    let x = u128::from(r[c].0[t]);
                    (a + x, b + x * x)
                });
                if n >= 2 {
                    isqrt((n * s2 - s * s) / (n * (n - 1))) as u64
                } else {
                    0
                }
            })
            .collect()
    }

    /// The value rows' bytes: deal by deal, candidate by candidate, four little-endian u64.
    pub fn value_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.values.len() * self.cands.len() * 32);
        for row in &self.values {
            assert_eq!(row.len(), self.cands.len(), "a value per candidate");
            for v in row {
                for x in v.0 {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
        out
    }

    /// Check the record's shape (sizes, codes, a value per deal and candidate).
    pub fn check(&self) -> Result<(), String> {
        if self.cands.is_empty() || self.cands.len() > MAX_CANDS {
            return Err(format!("{} candidates", self.cands.len()));
        }
        if self.pos.actions.len() > MAX_ACTIONS {
            return Err(format!("{} actions", self.pos.actions.len()));
        }
        if self.values.is_empty() {
            return Err("no deals".into());
        }
        if self.values.iter().any(|r| r.len() != self.cands.len()) {
            return Err("a deal without a value for every candidate".into());
        }
        if self
            .values
            .iter()
            .flatten()
            .any(|v| MoneyChance::from_raw(v.0).is_none())
        {
            return Err("a value that does not sum to 2^32".into());
        }
        Ok(())
    }
}

/// Integer square root (floor).
pub fn isqrt(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let mut x = 1u128 << (128 - n.leading_zeros()).div_ceil(2);
    loop {
        let y = (x + n / x) / 2;
        if y >= x {
            return x;
        }
        x = y;
    }
}

/// Per candidate, the plain mean of `values[deal][cand]` over the deals, in deal order.
pub fn means_of(values: &[Vec<MoneyChance>], n_cands: usize) -> Vec<MoneyChance> {
    (0..n_cands)
        .map(|c| {
            let terms: Vec<(u64, MoneyChance)> = values.iter().map(|r| (1, r[c])).collect();
            MoneyChance::mix(&terms)
        })
        .collect()
}

/// Per candidate, the mean of `values[deal][cand]` with deal `j` weighted `w[j]` (a card-reading
/// model's importance weights, docs/STUDENT.md §3 item 2), in deal order.
pub fn weighted_means_of(
    values: &[Vec<MoneyChance>],
    w: &[u64],
    n_cands: usize,
) -> Vec<MoneyChance> {
    assert_eq!(values.len(), w.len(), "a weight per deal");
    (0..n_cands)
        .map(|c| {
            let terms: Vec<(u64, MoneyChance)> =
                values.iter().zip(w).map(|(r, &w)| (w, r[c])).collect();
            MoneyChance::mix(&terms)
        })
        .collect()
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// A `.npy` header (format 1.0) for `descr` and `shape`, padded with spaces to `total` bytes (a
/// multiple of 64), or to the next multiple of 64 when `total` is 0.
fn npy_header(descr: &str, shape: &str, total: usize) -> Vec<u8> {
    let mut h = format!("{{'descr': {descr}, 'fortran_order': False, 'shape': {shape}, }}");
    let unpadded = 10 + h.len() + 1;
    let want = if total == 0 {
        unpadded.div_ceil(64) * 64
    } else {
        assert!(unpadded <= total, "the header fits");
        total
    };
    h.push_str(&" ".repeat(want - unpadded));
    h.push('\n');
    let mut out = Vec::with_capacity(want);
    out.extend_from_slice(b"\x93NUMPY");
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(h.len() as u16).to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    debug_assert_eq!(out.len(), want);
    out
}

/// The values file's header (a fixed 128 bytes, so it can be rewritten in place with the final
/// row count).
fn values_header(rows: u64) -> Vec<u8> {
    npy_header("'<u8'", &format!("({rows}, 4)"), 128)
}

/// The records file's header.
fn records_header(n: u64) -> Vec<u8> {
    npy_header(RECORD_DESCR, &format!("({n},)"), 0)
}

/// Parse a `.npy` header: (descr, shape, data offset).
fn parse_header(b: &[u8]) -> Result<(String, Vec<u64>, usize), String> {
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" || b[6] != 1 {
        return Err("not a version-1 .npy file".into());
    }
    let n = usize::from(u16::from_le_bytes([b[8], b[9]]));
    let h = std::str::from_utf8(b.get(10..10 + n).ok_or("short header")?)
        .map_err(|_| "header is not text")?;
    let d0 = h.find("'descr': ").ok_or("no descr")? + 9;
    let d1 = h.find(", 'fortran_order': False").ok_or("fortran order")?;
    let descr = h[d0..d1].to_string();
    let s0 = h.find("'shape': (").ok_or("no shape")? + 10;
    let s1 = s0 + h[s0..].find(')').ok_or("shape")?;
    let shape = h[s0..s1]
        .split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(|x| x.parse::<u64>().map_err(|_| format!("shape {x}")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((descr, shape, 10 + n))
}

fn put_u64s(out: &mut Vec<u8>, xs: impl IntoIterator<Item = u64>) {
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

/// The record's bytes (`values_at` and `values_sha256` as given).
fn record_bytes(r: &TeacherRecord, values_at: u64, values_sha: &[u8; 32], out: &mut Vec<u8>) {
    let start = out.len();
    let p = &r.pos;
    out.extend_from_slice(&p.sha256());
    out.extend_from_slice(&[
        r.source,
        r.trick,
        r.place,
        p.seat,
        r.stratum,
        r.kind,
        r.cands.len() as u8,
        p.actions.len() as u8,
    ]);
    out.extend_from_slice(&r.gen_seed.to_le_bytes());
    out.extend_from_slice(&r.index.to_le_bytes());
    let s = &p.start;
    out.push(s.dealer);
    out.extend_from_slice(&s.scores);
    out.extend(s.bonus_taken.iter().map(|&b| u8::from(b)));
    out.extend_from_slice(&s.games_won);
    out.push(s.game_index);
    out.extend_from_slice(&s.hand_index.to_le_bytes());
    out.extend_from_slice(&p.deck);
    for i in 0..MAX_ACTIONS {
        let x = p.actions.get(i).map_or(0, |&a| action_code(a));
        out.extend_from_slice(&x.to_le_bytes());
    }
    for i in 0..MAX_CANDS {
        let x = r.cands.get(i).map_or(0, |&c| cand_code(c));
        out.extend_from_slice(&x.to_le_bytes());
    }
    out.extend_from_slice(&r.dseed.to_le_bytes());
    out.extend_from_slice(&r.deal_from.to_le_bytes());
    out.extend_from_slice(&r.n_deals().to_le_bytes());
    out.extend_from_slice(&r.log2_worlds_q8.to_le_bytes());
    let means = r.means();
    for i in 0..MAX_CANDS {
        put_u64s(out, means.get(i).map_or([0; 4], |m| m.0));
    }
    let sds = r.own_sd();
    put_u64s(
        out,
        (0..MAX_CANDS).map(|i| sds.get(i).copied().unwrap_or(0)),
    );
    out.extend_from_slice(&values_at.to_le_bytes());
    out.extend_from_slice(values_sha);
    debug_assert_eq!(out.len() - start, RECORD_BYTES);
}

/// What a finished shard is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardInfo {
    /// The shard's key: the SHA-256 of the records file (lowercase hex).
    pub key: String,
    /// Records.
    pub records: u64,
    /// The records file's size.
    pub records_bytes: u64,
    /// The values file's SHA-256 (lowercase hex).
    pub values_sha256: String,
    /// Value rows.
    pub value_rows: u64,
    /// The values file's size.
    pub values_bytes: u64,
}

/// The values file beside a records file: `NAME.npy` → `NAME.values.npy`.
pub fn values_path(records: &Path) -> PathBuf {
    let s = records.to_string_lossy();
    PathBuf::from(match s.strip_suffix(".npy") {
        Some(stem) => format!("{stem}.values.npy"),
        None => format!("{s}.values.npy"),
    })
}

/// The sidecar beside a records file: `NAME.npy` → `NAME.npy.json`.
pub fn sidecar_path(records: &Path) -> PathBuf {
    PathBuf::from(format!("{}.json", records.to_string_lossy()))
}

fn part(p: &Path) -> PathBuf {
    PathBuf::from(format!("{}.part", p.to_string_lossy()))
}

fn sha256_file(p: &Path) -> Result<(String, u64), String> {
    let mut f = File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut n = 0u64;
    loop {
        let k = f
            .read(&mut buf)
            .map_err(|e| format!("{}: {e}", p.display()))?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        n += k as u64;
    }
    Ok((hex(&h.finalize()), n))
}

/// Writes a shard record by record, in order: the values stream to `NAME.values.npy.part` as the
/// records come, the records are kept (about 1.6 KB each) and written at [`ShardWriter::finish`],
/// which then rewrites the values file's header with its row count, hashes both files, writes the
/// sidecar and renames every file into place (so a shard under its final name is always whole).
pub struct ShardWriter {
    path: PathBuf,
    values: BufWriter<File>,
    records: Vec<u8>,
    n: u64,
    rows: u64,
}

impl ShardWriter {
    /// Start a shard whose records file will be `path` (`NAME.npy`).
    pub fn create(path: &Path) -> Result<ShardWriter, String> {
        let vp = part(&values_path(path));
        let mut values =
            BufWriter::new(File::create(&vp).map_err(|e| format!("{}: {e}", vp.display()))?);
        values
            .write_all(&values_header(0))
            .map_err(|e| e.to_string())?;
        Ok(ShardWriter {
            path: path.to_path_buf(),
            values,
            records: Vec::new(),
            n: 0,
            rows: 0,
        })
    }

    /// Append one record.
    pub fn push(&mut self, r: &TeacherRecord) -> Result<(), String> {
        r.check()?;
        let vb = r.value_bytes();
        let sha: [u8; 32] = Sha256::digest(&vb).into();
        self.values.write_all(&vb).map_err(|e| e.to_string())?;
        record_bytes(r, self.rows, &sha, &mut self.records);
        self.rows += (r.values.len() * r.cands.len()) as u64;
        self.n += 1;
        Ok(())
    }

    /// Records so far.
    pub fn len(&self) -> u64 {
        self.n
    }

    /// No records yet.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Finish the shard: `sidecar` (the caller's stamps, a JSON object) gets the format, the
    /// counts and both files' SHA-256 added and is written as `NAME.npy.json`.
    pub fn finish(self, sidecar: Value) -> Result<ShardInfo, String> {
        let ShardWriter {
            path,
            values,
            records,
            n,
            rows,
        } = self;
        let vp = values_path(&path);
        let mut f = values.into_inner().map_err(|e| e.to_string())?;
        f.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
        f.write_all(&values_header(rows))
            .map_err(|e| e.to_string())?;
        f.sync_all().map_err(|e| e.to_string())?;
        drop(f);
        let (values_sha256, values_bytes) = sha256_file(&part(&vp))?;
        let mut rb = records_header(n);
        rb.extend_from_slice(&records);
        let key = hex(&Sha256::digest(&rb));
        let rp = part(&path);
        std::fs::write(&rp, &rb).map_err(|e| format!("{}: {e}", rp.display()))?;
        let info = ShardInfo {
            key,
            records: n,
            records_bytes: rb.len() as u64,
            values_sha256,
            value_rows: rows,
            values_bytes,
        };
        let mut side = match sidecar {
            Value::Object(m) => m,
            Value::Null => serde_json::Map::new(),
            _ => return Err("the sidecar must be a JSON object".into()),
        };
        side.insert("format".into(), json!(FORMAT));
        side.insert("key".into(), json!(info.key));
        side.insert("records".into(), json!(info.records));
        side.insert("records_sha256".into(), json!(info.key));
        side.insert("records_bytes".into(), json!(info.records_bytes));
        side.insert("values_sha256".into(), json!(info.values_sha256));
        side.insert("value_rows".into(), json!(info.value_rows));
        side.insert("values_bytes".into(), json!(info.values_bytes));
        let sp = sidecar_path(&path);
        let text = serde_json::to_string_pretty(&Value::Object(side)).expect("JSON");
        std::fs::write(part(&sp), text + "\n").map_err(|e| e.to_string())?;
        // Values first, the records file (the shard's name) last: a shard present under its
        // final name is complete.
        for p in [&vp, &sp, &path] {
            std::fs::rename(part(p), p).map_err(|e| format!("{}: {e}", p.display()))?;
        }
        Ok(info)
    }
}

/// A record as read from a shard, without its values (read them with [`ShardReader::values`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordHead {
    /// The position.
    pub pos: RawPosition,
    /// The stored `pos_sha256` (checked against the position when read).
    pub pos_sha256: [u8; 32],
    /// Source.
    pub source: u8,
    /// Generator seed.
    pub gen_seed: u64,
    /// Index in the generator's stream.
    pub index: u64,
    /// Trick.
    pub trick: u8,
    /// Place in the trick.
    pub place: u8,
    /// Stratum.
    pub stratum: u8,
    /// Kind.
    pub kind: u8,
    /// Candidates.
    pub cands: Vec<Cand>,
    /// Decision seed.
    pub dseed: u64,
    /// First deal.
    pub deal_from: u32,
    /// Deals.
    pub n_deals: u32,
    /// World count (log2, ×256).
    pub log2_worlds_q8: u16,
    /// The stored means (checked against the values when they are read).
    pub means: Vec<MoneyChance>,
    /// The stored own-pair SDs.
    pub own_sd: Vec<u64>,
    /// First row in the values file.
    pub values_at: u64,
    /// SHA-256 of the record's value rows.
    pub values_sha256: [u8; 32],
}

struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> &[u8] {
        let s = &self.b[self.at..self.at + n];
        self.at += n;
        s
    }
    fn u8(&mut self) -> u8 {
        self.take(1)[0]
    }
    fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.take(2).try_into().expect("2"))
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().expect("4"))
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take(8).try_into().expect("8"))
    }
    fn arr<const N: usize>(&mut self) -> [u8; N] {
        self.take(N).try_into().expect("N")
    }
}

fn parse_record(b: &[u8]) -> Result<RecordHead, String> {
    let mut c = Cursor { b, at: 0 };
    let pos_sha256 = c.arr::<32>();
    let [
        source,
        trick,
        place,
        seat,
        stratum,
        kind,
        n_cands,
        n_actions,
    ] = c.arr::<8>();
    let gen_seed = c.u64();
    let index = c.u64();
    let dealer = c.u8();
    let scores = c.arr::<4>();
    let bonus = c.arr::<4>();
    let games_won = c.arr::<4>();
    let game_index = c.u8();
    let hand_index = c.u32();
    let deck = c.arr::<DECK_SIZE>();
    let codes: Vec<u16> = (0..MAX_ACTIONS).map(|_| c.u16()).collect();
    let ccodes: Vec<u16> = (0..MAX_CANDS).map(|_| c.u16()).collect();
    let dseed = c.u64();
    let deal_from = c.u32();
    let n_deals = c.u32();
    let log2_worlds_q8 = c.u16();
    let means: Vec<[u64; 4]> = (0..MAX_CANDS)
        .map(|_| [c.u64(), c.u64(), c.u64(), c.u64()])
        .collect();
    let sds: Vec<u64> = (0..MAX_CANDS).map(|_| c.u64()).collect();
    let values_at = c.u64();
    let values_sha256 = c.arr::<32>();
    let (n_actions, n_cands) = (usize::from(n_actions), usize::from(n_cands));
    if n_actions > MAX_ACTIONS || n_cands == 0 || n_cands > MAX_CANDS {
        return Err("action or candidate count out of range".into());
    }
    let actions = codes[..n_actions]
        .iter()
        .map(|&x| action_from_code(x).ok_or_else(|| format!("action code {x}")))
        .collect::<Result<Vec<_>, _>>()?;
    let cands = ccodes[..n_cands]
        .iter()
        .map(|&x| cand_from_code(x).ok_or_else(|| format!("candidate code {x}")))
        .collect::<Result<Vec<_>, _>>()?;
    let means = means[..n_cands]
        .iter()
        .map(|&m| MoneyChance::from_raw(m).ok_or("a mean that does not sum to 2^32"))
        .collect::<Result<Vec<_>, _>>()?;
    let pos = RawPosition {
        start: HandStartState {
            dealer,
            scores,
            bonus_taken: bonus.map(|b| b != 0),
            games_won,
            hand_index,
            game_index,
        },
        deck,
        actions,
        seat,
    };
    if pos.sha256() != pos_sha256 {
        return Err("pos_sha256 is not the position's".into());
    }
    Ok(RecordHead {
        pos,
        pos_sha256,
        source,
        gen_seed,
        index,
        trick,
        place,
        stratum,
        kind,
        cands,
        dseed,
        deal_from,
        n_deals,
        log2_worlds_q8,
        means,
        own_sd: sds[..n_cands].to_vec(),
        values_at,
        values_sha256,
    })
}

/// A shard opened for reading: every record's head, the sidecar, and the values file.
pub struct ShardReader {
    /// The records file.
    pub path: PathBuf,
    /// The shard's key (the records file's SHA-256).
    pub key: String,
    /// The sidecar.
    pub sidecar: Value,
    /// The records, in order.
    pub records: Vec<RecordHead>,
    values: File,
    values_data: u64,
    value_rows: u64,
}

impl ShardReader {
    /// Open a shard: read and check the records file (its type, every record's codes and
    /// position identity) and the sidecar (format, key, the values file's size and SHA-256 are
    /// those recorded).
    pub fn open(path: &Path) -> Result<ShardReader, String> {
        let rb = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let key = hex(&Sha256::digest(&rb));
        let (descr, shape, off) = parse_header(&rb)?;
        if descr != RECORD_DESCR {
            return Err(format!("{}: not a teacher records file", path.display()));
        }
        let n = *shape.first().ok_or("shape")? as usize;
        if shape.len() != 1 || rb.len() != off + n * RECORD_BYTES {
            return Err(format!("{}: size", path.display()));
        }
        let records = (0..n)
            .map(|i| parse_record(&rb[off + i * RECORD_BYTES..off + (i + 1) * RECORD_BYTES]))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let sp = sidecar_path(path);
        let sidecar: Value = serde_json::from_str(
            &std::fs::read_to_string(&sp).map_err(|e| format!("{}: {e}", sp.display()))?,
        )
        .map_err(|e| format!("{}: {e}", sp.display()))?;
        if sidecar["format"] != FORMAT {
            return Err(format!("{}: format is not {FORMAT}", sp.display()));
        }
        if sidecar["key"] != key.as_str() {
            return Err(format!(
                "{}: the records file's SHA-256 is {key}, the sidecar says {}",
                path.display(),
                sidecar["key"]
            ));
        }
        let vp = values_path(path);
        let mut values = File::open(&vp).map_err(|e| format!("{}: {e}", vp.display()))?;
        let mut head = vec![0u8; 128];
        values
            .read_exact(&mut head)
            .map_err(|e| format!("{}: {e}", vp.display()))?;
        let (vd, vs, voff) = parse_header(&head)?;
        if vd != "'<u8'" || vs.len() != 2 || vs[1] != 4 {
            return Err(format!("{}: not a values file", vp.display()));
        }
        let size = values.metadata().map_err(|e| e.to_string())?.len();
        if size != voff as u64 + 32 * vs[0] || sidecar["values_bytes"] != size {
            return Err(format!("{}: size", vp.display()));
        }
        Ok(ShardReader {
            path: path.to_path_buf(),
            key,
            sidecar,
            records,
            values,
            values_data: voff as u64,
            value_rows: vs[0],
        })
    }

    /// The values of record `i` (`[deal][candidate]`), checked against the record's SHA-256 and
    /// its stored means.
    pub fn values(&mut self, i: usize) -> Result<Vec<Vec<MoneyChance>>, String> {
        let r = &self.records[i];
        let n = r.cands.len();
        let rows = u64::from(r.n_deals) * n as u64;
        if r.values_at + rows > self.value_rows {
            return Err(format!("record {i}: values beyond the file"));
        }
        let mut b = vec![0u8; 32 * rows as usize];
        self.values
            .seek(SeekFrom::Start(self.values_data + 32 * r.values_at))
            .map_err(|e| e.to_string())?;
        self.values.read_exact(&mut b).map_err(|e| e.to_string())?;
        let sha: [u8; 32] = Sha256::digest(&b).into();
        if sha != r.values_sha256 {
            return Err(format!(
                "{} record {i}: its values are not the ones recorded (SHA-256)",
                self.path.display()
            ));
        }
        let mut out = Vec::with_capacity(r.n_deals as usize);
        for d in 0..r.n_deals as usize {
            let mut row = Vec::with_capacity(n);
            for c in 0..n {
                let at = 32 * (d * n + c);
                let v: [u64; 4] = core::array::from_fn(|t| {
                    u64::from_le_bytes(b[at + 8 * t..at + 8 * t + 8].try_into().expect("8"))
                });
                row.push(MoneyChance::from_raw(v).ok_or("a value that does not sum to 2^32")?);
            }
            out.push(row);
        }
        if means_of(&out, n) != r.means {
            return Err(format!("record {i}: stored means differ from the values"));
        }
        Ok(out)
    }

    /// The whole record `i`, values included.
    pub fn record(&mut self, i: usize) -> Result<TeacherRecord, String> {
        let values = self.values(i)?;
        let h = &self.records[i];
        Ok(TeacherRecord {
            pos: h.pos.clone(),
            source: h.source,
            gen_seed: h.gen_seed,
            index: h.index,
            trick: h.trick,
            place: h.place,
            stratum: h.stratum,
            kind: h.kind,
            cands: h.cands.clone(),
            dseed: h.dseed,
            deal_from: h.deal_from,
            log2_worlds_q8: h.log2_worlds_q8,
            values,
        })
    }

    /// Check the values file's SHA-256 against the sidecar (reads the whole file).
    pub fn verify_values_file(&self) -> Result<(), String> {
        let (sha, _) = sha256_file(&values_path(&self.path))?;
        if self.sidecar["values_sha256"] != sha.as_str() {
            return Err(format!(
                "{}: the values file's SHA-256 is not the sidecar's",
                self.path.display()
            ));
        }
        Ok(())
    }
}

/// The stamps that must agree for data to be used together (docs/STUDENT.md §3): a shard's value
/// of each, as text.
pub const STAMPS: [&str; 7] = [
    "teacher",
    "equity_sha256",
    "rules_version",
    "values_version",
    "sampler",
    "candidates",
    "trick",
];

/// The sidecar's stamp `k` as text (`null` when absent).
pub fn stamp(sidecar: &Value, k: &str) -> String {
    match &sidecar[k] {
        Value::String(s) => s.clone(),
        v => v.to_string(),
    }
}

/// Check that every shard's stamps equal the first's, except those in `allow`; the first
/// difference is the error.
pub fn check_stamps(sidecars: &[(&str, &Value)], allow: &[String]) -> Result<(), String> {
    let Some((first_name, first)) = sidecars.first() else {
        return Ok(());
    };
    for (name, s) in &sidecars[1..] {
        for k in STAMPS {
            if allow.iter().any(|a| a == k) {
                continue;
            }
            if stamp(first, k) != stamp(s, k) {
                return Err(format!(
                    "stamp {k}: {first_name} has {}, {name} has {} (allow it with --allow-stamp {k})",
                    stamp(first, k),
                    stamp(s, k)
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::game::{Action, Match, Rules};

    fn record(seed: u64, n_deals: usize) -> TeacherRecord {
        let start = HandStartState {
            dealer: (seed % 8) as u8,
            scores: [0, 5, 11, 20],
            bonus_taken: [false, false, true, false],
            games_won: [0, 1, 0, 0],
            hand_index: 3,
            game_index: 1,
        };
        let mut deck: [u8; DECK_SIZE] = core::array::from_fn(|c| c as u8);
        engine::rng::Rng::new(seed).shuffle(&mut deck);
        let mut m = Match::from_hand_start(start, deck, 0, Rules::HOUSE).expect("valid");
        let mut actions = Vec::new();
        while m.public().trick_no < 2 {
            let a = *m.legal_actions().last().expect("legal");
            m.apply(a).expect("legal");
            actions.push(a);
        }
        let seat = m.to_act().expect("to act");
        let cands: Vec<Cand> = m.legal_actions().into_iter().map(|a| (a, None)).collect();
        let mut rng = engine::rng::Rng::new(seed ^ 77);
        let values = (0..n_deals)
            .map(|_| {
                cands
                    .iter()
                    .map(|_| {
                        MoneyChance::from_weights(core::array::from_fn(|_| {
                            u128::from(rng.next_u64() >> 20) + 1
                        }))
                    })
                    .collect()
            })
            .collect();
        TeacherRecord {
            pos: RawPosition {
                start,
                deck,
                actions,
                seat,
            },
            source: SOURCE_RULES,
            gen_seed: seed,
            index: seed * 3,
            trick: 2,
            place: 0,
            stratum: 2,
            kind: KIND_PLAY,
            cands,
            dseed: seed ^ 0xD,
            deal_from: 5,
            log2_worlds_q8: 1234,
            values,
        }
    }

    #[test]
    fn a_shard_round_trips_and_its_key_covers_the_values() {
        let dir = std::env::temp_dir().join(format!("student-record-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("t.npy");
        let recs: Vec<TeacherRecord> = (1..6).map(|s| record(s, 3 + s as usize)).collect();
        let mut w = ShardWriter::create(&path).expect("create");
        for r in &recs {
            w.push(r).expect("push");
        }
        let info = w.finish(json!({ "trick": 2 })).expect("finish");
        assert_eq!(info.records, 5);
        assert!(!part(&path).exists() && path.exists() && values_path(&path).exists());
        let mut rd = ShardReader::open(&path).expect("open");
        rd.verify_values_file().expect("values file");
        assert_eq!(rd.key, info.key);
        for (i, r) in recs.iter().enumerate() {
            let got = rd.record(i).expect("record");
            assert_eq!(&got, r);
            assert_eq!(rd.records[i].means, r.means());
            assert_eq!(rd.records[i].own_sd, r.own_sd());
        }
        // The same records give the same bytes and key.
        let path2 = dir.join("u.npy");
        let mut w = ShardWriter::create(&path2).expect("create");
        for r in &recs {
            w.push(r).expect("push");
        }
        let info2 = w.finish(json!({ "trick": 2 })).expect("finish");
        assert_eq!(info2.key, info.key);
        assert_eq!(info2.values_sha256, info.values_sha256);
        // A changed value is caught on read, though the records file is unchanged.
        let vp = values_path(&path);
        let mut vb = std::fs::read(&vp).expect("values");
        let last = vb.len() - 9;
        vb[last] ^= 1;
        std::fs::write(&vp, &vb).expect("write");
        let mut rd = ShardReader::open(&path).expect("open");
        assert!(rd.values(4).is_err());
        assert!(rd.verify_values_file().is_err());
        assert!(rd.values(0).is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn records_are_their_documented_size_and_numpy_type() {
        assert_eq!(RECORD_BYTES, 1625);
        let h = records_header(7);
        assert_eq!(h.len() % 64, 0);
        let (d, s, off) = parse_header(&h).expect("parse");
        assert_eq!((d.as_str(), s, off), (RECORD_DESCR, vec![7], h.len()));
        let h = values_header(12_345_678_901);
        assert_eq!(h.len(), 128);
        assert_eq!(parse_header(&h).expect("parse").1, vec![12_345_678_901, 4]);
    }

    #[test]
    fn stamps_must_agree_unless_allowed() {
        let a = json!({ "teacher": "hybrid:abc", "equity_sha256": "e", "trick": 2 });
        let b = json!({ "teacher": "hybrid:def", "equity_sha256": "e", "trick": 2 });
        assert!(check_stamps(&[("a", &a), ("a2", &a)], &[]).is_ok());
        let e = check_stamps(&[("a", &a), ("b", &b)], &[]).unwrap_err();
        assert!(e.contains("teacher"), "{e}");
        assert!(check_stamps(&[("a", &a), ("b", &b)], &["teacher".to_string()]).is_ok());
        let _ = Action::Play(0);
    }
}
