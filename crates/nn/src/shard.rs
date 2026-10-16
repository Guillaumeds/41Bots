//! Label shards (docs/M3-HYBRID-SPEC.md §5; PLAN §12.7): one `.npy` file [NumPy's standard
//! array file] of fixed little-endian records, readable (and memory-mappable) by `numpy.load`:
//!
//! ```text
//! dtype [('active', '<u2', (96,)), ('dense', '<i4', (100,)), ('value', '<u8', (4,)),
//!        ('source', '|u1'), ('trick', '|u1')]
//! ```
//! - `active`: the encoder's active binary feature indices, ascending, then 65535 (padding);
//! - `dense`: the encoder's dense values (`log2` of money chances / 32, units of 2^-24);
//! - `value`: the target, the exact solver's money chance of the position in the encoder's
//!   rotated team order (four u64 in units of 2^-32, summing to exactly 2^32);
//! - `source`: where the position came from ([`SOURCE_RANDOM`], [`SOURCE_RULES`],
//!   [`SOURCE_HARVEST`]);
//! - `trick`: the trick whose start the position is (2..=5).
//!
//! Records are packed (626 bytes each, no padding). The header is format version 1.0, padded so
//! the data starts on a 64-byte boundary.

use equity::MoneyChance;

use crate::encode::{Features, MAX_ACTIVE, N_DENSE, PAD};

/// Random legal play from yardstick decision roots inside belief-sampled worlds.
pub const SOURCE_RANDOM: u8 = 1;
/// `rules-v1` self-play.
pub const SOURCE_RULES: u8 = 2;
/// Frontier positions harvested from the hybrid's own searches (second round of a rung).
pub const SOURCE_HARVEST: u8 = 3;
/// Synthetic positions for tests (`nn-tool fixture`).
pub const SOURCE_FIXTURE: u8 = 9;

/// Bytes per record.
pub const ROW_BYTES: usize = 2 * MAX_ACTIVE + 4 * N_DENSE + 32 + 2;

const MAGIC: &[u8; 6] = b"\x93NUMPY";

/// One labelled position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Row {
    /// The encoded position.
    pub features: Features,
    /// The target in the encoder's rotated team order.
    pub value: MoneyChance,
    /// Where it came from.
    pub source: u8,
    /// The trick whose start it is.
    pub trick: u8,
}

/// The record type as NumPy writes it.
pub fn descr() -> String {
    format!(
        "[('active', '<u2', ({MAX_ACTIVE},)), ('dense', '<i4', ({N_DENSE},)), \
         ('value', '<u8', (4,)), ('source', '|u1'), ('trick', '|u1')]"
    )
}

fn header(n: usize) -> Vec<u8> {
    let mut h = format!(
        "{{'descr': {}, 'fortran_order': False, 'shape': ({n},), }}",
        descr()
    );
    let unpadded = 10 + h.len() + 1;
    let pad = (64 - unpadded % 64) % 64;
    h.push_str(&" ".repeat(pad));
    h.push('\n');
    let len = u16::try_from(h.len()).expect("header fits a u16");
    let mut out = Vec::with_capacity(10 + h.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    debug_assert_eq!(out.len() % 64, 0);
    out
}

/// One record's bytes.
pub fn row_bytes(r: &Row, out: &mut Vec<u8>) {
    for a in r.features.active {
        out.extend_from_slice(&a.to_le_bytes());
    }
    for d in r.features.dense {
        out.extend_from_slice(&d.to_le_bytes());
    }
    for v in r.value.0 {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.push(r.source);
    out.push(r.trick);
}

/// The `.npy` bytes of a shard.
pub fn to_npy(rows: &[Row]) -> Vec<u8> {
    let mut out = header(rows.len());
    out.reserve(rows.len() * ROW_BYTES);
    for r in rows {
        row_bytes(r, &mut out);
    }
    out
}

/// Read a shard written by [`to_npy`] (or by NumPy with the same record type).
pub fn from_npy(bytes: &[u8]) -> Result<Vec<Row>, String> {
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err("not a .npy file".into());
    }
    let (hlen, start) = match bytes[6] {
        1 => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        v => return Err(format!("unsupported .npy version {v}")),
    };
    let h = core::str::from_utf8(bytes.get(start..start + hlen).ok_or("truncated header")?)
        .map_err(|_| "header is not text")?;
    let compact: String = h.chars().filter(|c| !c.is_whitespace()).collect();
    let want: String = descr().chars().filter(|c| !c.is_whitespace()).collect();
    if !compact.contains(&format!("'descr':{want}")) || !compact.contains("'fortran_order':False") {
        return Err(format!("not a label shard (record type): {h}"));
    }
    let shape = compact
        .split("'shape':(")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .ok_or("no shape")?;
    let n: usize = shape
        .trim_end_matches(',')
        .parse()
        .map_err(|_| format!("bad shape {shape}"))?;
    let body = &bytes[start + hlen..];
    if body.len() != n * ROW_BYTES {
        return Err(format!(
            "{} data bytes for {n} records of {ROW_BYTES}",
            body.len()
        ));
    }
    let mut rows = Vec::with_capacity(n);
    for rec in body.as_chunks::<ROW_BYTES>().0 {
        let u16_at = |i: usize| u16::from_le_bytes([rec[i], rec[i + 1]]);
        let mut f = Features {
            active: [PAD; MAX_ACTIVE],
            n_active: 0,
            dense: [0; N_DENSE],
        };
        for k in 0..MAX_ACTIVE {
            f.active[k] = u16_at(2 * k);
        }
        f.n_active = f.active.iter().take_while(|&&a| a != PAD).count() as u8;
        if f.active[usize::from(f.n_active)..]
            .iter()
            .any(|&a| a != PAD)
        {
            return Err("active indices after the padding".into());
        }
        for k in 0..N_DENSE {
            let i = 2 * MAX_ACTIVE + 4 * k;
            f.dense[k] = i32::from_le_bytes([rec[i], rec[i + 1], rec[i + 2], rec[i + 3]]);
        }
        let o = 2 * MAX_ACTIVE + 4 * N_DENSE;
        let v: [u64; 4] = core::array::from_fn(|t| {
            u64::from_le_bytes(rec[o + 8 * t..o + 8 * t + 8].try_into().expect("8 bytes"))
        });
        rows.push(Row {
            features: f,
            value: MoneyChance::from_raw(v).ok_or("a value that does not sum to 2^32")?,
            source: rec[o + 32],
            trick: rec[o + 33],
        });
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut f = Features {
            active: [PAD; MAX_ACTIVE],
            n_active: 3,
            dense: [-7; N_DENSE],
        };
        f.active[..3].copy_from_slice(&[1, 400, 808]);
        let rows = vec![
            Row {
                features: f,
                value: MoneyChance::from_weights([1, 2, 3, 4]),
                source: SOURCE_RANDOM,
                trick: 4,
            },
            Row {
                features: f,
                value: MoneyChance::certain(2),
                source: SOURCE_RULES,
                trick: 3,
            },
        ];
        let b = to_npy(&rows);
        assert_eq!(b.len() % 2, 0);
        assert_eq!(from_npy(&b).expect("reads"), rows);
        assert!(from_npy(&b[..b.len() - 1]).is_err());
    }
}
