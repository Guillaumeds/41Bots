//! Feature shards (docs/STUDENT.md §5, format `41bots-student-features-v1`): the teacher's
//! records encoded for training by this build's encoder (`crate::encode`), one `.npy` record per
//! decision, readable and memory-mappable by `numpy.load`, and a JSON sidecar (the encoder's name,
//! hash and sizes, the teacher shards' keys and stamps). Python reads only these; it never encodes.
//!
//! ```text
//! dtype [('view', '<u2', (448,)), ('dense', '<i4', (100,)), ('n_cands', '|u1'),
//!        ('cand', '<u2', (32, 4)), ('target', '<u8', (32, 4)), ('sd', '<u8', (32,)),
//!        ('n_deals', '<u4'), ('trick', '|u1'), ('place', '|u1'), ('stratum', '|u1'),
//!        ('source', '|u1'), ('kind', '|u1'), ('pos_key', '<u8')]
//! ```
//! - `view`: the view's active binary features, ascending, then 65535;
//! - `dense`: the dense block;
//! - `cand[c]`: candidate `c`'s active features (indices in the same space, after the view's),
//!   then 65535; rows past `n_cands` are all 65535;
//! - `target[c]`: the teacher's mean money chance of candidate `c` over its deals, in the
//!   network's rotated pair order (the deciding pair first), four u64 summing to 2^32 (0 past
//!   `n_cands`);
//! - `sd[c]`: the sample SD over the deals of the deciding pair's value (units of 2^-32);
//! - `pos_key`: the first 8 bytes (little-endian) of the position's SHA-256: the validation split
//!   is by position, never by row.

use equity::MoneyChance;
use nn::encode::N_DENSE;

use crate::encode::{K_MAX, PAD, V_MAX, ViewFeatures};
use crate::record::MAX_CANDS;

/// The format name.
pub const FORMAT: &str = "41bots-student-features-v1";

/// The record type, as NumPy writes it.
pub fn descr() -> String {
    format!(
        "[('view', '<u2', ({V_MAX},)), ('dense', '<i4', ({N_DENSE},)), ('n_cands', '|u1'), \
         ('cand', '<u2', ({MAX_CANDS}, {K_MAX})), ('target', '<u8', ({MAX_CANDS}, 4)), \
         ('sd', '<u8', ({MAX_CANDS},)), ('n_deals', '<u4'), ('trick', '|u1'), ('place', '|u1'), \
         ('stratum', '|u1'), ('source', '|u1'), ('kind', '|u1'), ('pos_key', '<u8')]"
    )
}

/// Bytes per record.
pub const ROW_BYTES: usize = 2 * V_MAX
    + 4 * N_DENSE
    + 1
    + 2 * MAX_CANDS * K_MAX
    + 32 * MAX_CANDS
    + 8 * MAX_CANDS
    + 4
    + 5
    + 8;

/// One decision, encoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeatureRow {
    /// The view.
    pub view: ViewFeatures,
    /// Each candidate's active features.
    pub cands: Vec<Vec<u16>>,
    /// Each candidate's target, rotated (the deciding pair first).
    pub targets: Vec<MoneyChance>,
    /// Each candidate's own-pair SD over the deals.
    pub sd: Vec<u64>,
    /// Deals behind the targets.
    pub n_deals: u32,
    /// Trick.
    pub trick: u8,
    /// Place in the trick.
    pub place: u8,
    /// Stratum.
    pub stratum: u8,
    /// Source.
    pub source: u8,
    /// Kind.
    pub kind: u8,
    /// The position key.
    pub pos_key: u64,
}

/// The `.npy` header of `n` records.
pub fn header(n: u64) -> Vec<u8> {
    let mut h = format!(
        "{{'descr': {}, 'fortran_order': False, 'shape': ({n},), }}",
        descr()
    );
    let unpadded = 10 + h.len() + 1;
    let pad = (64 - unpadded % 64) % 64;
    h.push_str(&" ".repeat(pad));
    h.push('\n');
    let mut out = Vec::with_capacity(10 + h.len());
    out.extend_from_slice(b"\x93NUMPY");
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(h.len() as u16).to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    out
}

/// One record's bytes.
pub fn row_bytes(r: &FeatureRow, out: &mut Vec<u8>) {
    let start = out.len();
    assert!(
        r.view.active.len() <= V_MAX,
        "view features within the room"
    );
    assert!(
        !r.cands.is_empty() && r.cands.len() <= MAX_CANDS,
        "1..=32 candidates"
    );
    assert_eq!(r.cands.len(), r.targets.len());
    assert_eq!(r.cands.len(), r.sd.len());
    for i in 0..V_MAX {
        out.extend_from_slice(&r.view.active.get(i).copied().unwrap_or(PAD).to_le_bytes());
    }
    for d in r.view.dense {
        out.extend_from_slice(&d.to_le_bytes());
    }
    out.push(r.cands.len() as u8);
    for c in 0..MAX_CANDS {
        let cand = r.cands.get(c).map_or(&[][..], |v| &v[..]);
        assert!(cand.len() <= K_MAX);
        for k in 0..K_MAX {
            out.extend_from_slice(&cand.get(k).copied().unwrap_or(PAD).to_le_bytes());
        }
    }
    for c in 0..MAX_CANDS {
        for x in r.targets.get(c).map_or([0; 4], |m| m.0) {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    for c in 0..MAX_CANDS {
        out.extend_from_slice(&r.sd.get(c).copied().unwrap_or(0).to_le_bytes());
    }
    out.extend_from_slice(&r.n_deals.to_le_bytes());
    out.extend_from_slice(&[r.trick, r.place, r.stratum, r.source, r.kind]);
    out.extend_from_slice(&r.pos_key.to_le_bytes());
    debug_assert_eq!(out.len() - start, ROW_BYTES);
}

/// Read a feature shard's records (the team of each view is not stored: it is 0 in what this
/// returns, so the network's outputs read back in the rotated order).
pub fn read(bytes: &[u8]) -> Result<Vec<FeatureRow>, String> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let hl = usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
    let h = std::str::from_utf8(&bytes[10..10 + hl]).map_err(|_| "header")?;
    if !h.contains(&descr()) {
        return Err("not a student feature shard".into());
    }
    let off = 10 + hl;
    if !(bytes.len() - off).is_multiple_of(ROW_BYTES) {
        return Err("size".into());
    }
    let n = (bytes.len() - off) / ROW_BYTES;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let b = &bytes[off + i * ROW_BYTES..off + (i + 1) * ROW_BYTES];
        let mut at = 0;
        let u16s = |k: usize, at: &mut usize| -> Vec<u16> {
            let v = (0..k)
                .map(|j| u16::from_le_bytes([b[*at + 2 * j], b[*at + 2 * j + 1]]))
                .collect();
            *at += 2 * k;
            v
        };
        let view: Vec<u16> = u16s(V_MAX, &mut at)
            .into_iter()
            .filter(|&x| x != PAD)
            .collect();
        let dense: [i32; N_DENSE] = core::array::from_fn(|d| {
            i32::from_le_bytes(b[at + 4 * d..at + 4 * d + 4].try_into().expect("4"))
        });
        at += 4 * N_DENSE;
        let nc = usize::from(b[at]);
        at += 1;
        let mut cands = Vec::with_capacity(nc);
        for c in 0..MAX_CANDS {
            let v: Vec<u16> = u16s(K_MAX, &mut at)
                .into_iter()
                .filter(|&x| x != PAD)
                .collect();
            if c < nc {
                cands.push(v);
            }
        }
        let u64_at = |at: usize| u64::from_le_bytes(b[at..at + 8].try_into().expect("8"));
        let mut targets = Vec::with_capacity(nc);
        for c in 0..MAX_CANDS {
            let m: [u64; 4] = core::array::from_fn(|t| u64_at(at + 32 * c + 8 * t));
            if c < nc {
                targets.push(MoneyChance::from_raw(m).ok_or("a target that is not a chance")?);
            }
        }
        at += 32 * MAX_CANDS;
        let sd: Vec<u64> = (0..nc).map(|c| u64_at(at + 8 * c)).collect();
        at += 8 * MAX_CANDS;
        let n_deals = u32::from_le_bytes(b[at..at + 4].try_into().expect("4"));
        at += 4;
        let [trick, place, stratum, source, kind] = b[at..at + 5].try_into().expect("5");
        at += 5;
        let pos_key = u64_at(at);
        out.push(FeatureRow {
            view: ViewFeatures {
                active: view,
                dense,
                team: 0,
            },
            cands,
            targets,
            sd,
            n_deals,
            trick,
            place,
            stratum,
            source,
            kind,
            pos_key,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_round_trip() {
        let r = FeatureRow {
            view: ViewFeatures {
                active: vec![1, 7, 300],
                dense: core::array::from_fn(|d| -(d as i32) * 1000),
                team: 0,
            },
            cands: vec![vec![3100, 3150], vec![3101]],
            targets: vec![MoneyChance::even(), MoneyChance::certain(2)],
            sd: vec![5, 6],
            n_deals: 320,
            trick: 2,
            place: 3,
            stratum: 4,
            source: 1,
            kind: 0,
            pos_key: 0xDEAD_BEEF,
        };
        let mut b = header(2);
        assert_eq!(b.len() % 64, 0);
        row_bytes(&r, &mut b);
        row_bytes(&r, &mut b);
        assert_eq!(read(&b).expect("reads"), vec![r.clone(), r]);
    }
}
