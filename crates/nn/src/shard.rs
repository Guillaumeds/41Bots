//! Label shards (docs/M3-HYBRID-SPEC.md §5; PLAN §12.7): one `.npy` file [NumPy's standard
//! array file] of fixed little-endian records, readable (and memory-mappable) by `numpy.load`:
//!
//! ```text
//! dtype [('active', '<u2', (96,)), ('dense', '<i4', (100,)), ('value', '<u8', (4,)),
//!        ('source', '|u1'), ('trick', '|u1'), ('label', '|u1'), ('mask', '<u8'),
//!        ('outcomes', '<u8', (V, 4))]
//! ```
//! - `active`: the encoder's active binary feature indices, ascending, then 65535 (padding);
//! - `dense`: the encoder's dense values (`log2` of money chances / 32, units of 2^-24);
//! - `value`: the target, the exact solver's money chance of the position in the encoder's
//!   rotated team order (four u64 in units of 2^-32, summing to exactly 2^32);
//! - `source`: where the position came from ([`SOURCE_RANDOM`], [`SOURCE_RULES`],
//!   [`SOURCE_HARVEST`]);
//! - `trick`: the trick whose start the position is (2..=5);
//! - `label`, `mask`, `outcomes`: the outcome head's target (spec §6a, `crate::outcome`): the
//!   label (first possible outcome whose money vector equals `value`), the possible outcomes
//!   (bit `o`), and every outcome's exact money vector (an even split where impossible); `V` is
//!   the vocabulary size of the trick (8, 14, 24, 39 for tricks 5, 4, 3, 2).
//!
//! Records are packed (no padding). The header is format version 1.0, padded so
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

/// Bytes per record with `v` outcomes.
pub const fn row_bytes_for(v: usize) -> usize {
    2 * MAX_ACTIVE + 4 * N_DENSE + 32 + 2 + 1 + 8 + 32 * v
}

const MAGIC: &[u8; 6] = b"\x93NUMPY";

/// One labelled position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    /// The encoded position.
    pub features: Features,
    /// The target in the encoder's rotated team order.
    pub value: MoneyChance,
    /// Where it came from.
    pub source: u8,
    /// The trick whose start it is.
    pub trick: u8,
    /// The outcome label.
    pub label: u8,
    /// The possible outcomes.
    pub mask: u64,
    /// Every outcome's money vector.
    pub outcomes: Vec<MoneyChance>,
}

impl Row {
    /// The row's race class ([`crate::outcome::race_class`]) from what the row holds: its
    /// possible outcomes (`mask`) and the binary features of the teams with a game won, both in
    /// the encoder's rotated team order (the trainer computes it the same way, `python/nn`).
    pub fn race_class(&self) -> u8 {
        let active = self.features.active();
        let has_game =
            core::array::from_fn(|t| active.contains(&((crate::encode::OFF_GAMES + t) as u16)));
        crate::outcome::race_class(self.mask, has_game)
    }
}

/// The record type with `v` outcomes, as NumPy writes it.
pub fn descr(v: usize) -> String {
    format!(
        "[('active', '<u2', ({MAX_ACTIVE},)), ('dense', '<i4', ({N_DENSE},)), \
         ('value', '<u8', (4,)), ('source', '|u1'), ('trick', '|u1'), ('label', '|u1'), \
         ('mask', '<u8'), ('outcomes', '<u8', ({v}, 4))]"
    )
}

/// The `.npy` header of a shard of `n` rows with `v` outcomes (what [`to_npy`] writes before the
/// records), for writers that stream the records after it.
pub fn header(n: usize, v: usize) -> Vec<u8> {
    let mut h = format!(
        "{{'descr': {}, 'fortran_order': False, 'shape': ({n},), }}",
        descr(v)
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
    out.push(r.label);
    out.extend_from_slice(&r.mask.to_le_bytes());
    for m in &r.outcomes {
        for x in m.0 {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
}

/// The `.npy` bytes of a shard whose rows all have `v` outcomes.
///
/// # Panics
/// If a row has another number of outcomes.
pub fn to_npy(rows: &[Row], v: usize) -> Vec<u8> {
    let mut out = header(rows.len(), v);
    out.reserve(rows.len() * row_bytes_for(v));
    for r in rows {
        assert_eq!(r.outcomes.len(), v, "every row has the trick's outcomes");
        row_bytes(r, &mut out);
    }
    out
}

/// A shard's header, from the start of its file (the whole header; the records may follow or
/// not): its number of records, its number of outcomes and where its records begin.
///
/// # Errors
/// If the bytes do not start a label shard's `.npy` file.
pub fn parse_header(bytes: &[u8]) -> Result<(usize, usize, usize), String> {
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err("not a .npy file".into());
    }
    let (hlen, start) = match bytes[6] {
        1 => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
        2 | 3 if bytes.len() >= 12 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        v => return Err(format!("unsupported .npy version {v}")),
    };
    let h = core::str::from_utf8(bytes.get(start..start + hlen).ok_or("truncated header")?)
        .map_err(|_| "header is not text")?;
    let compact: String = h.chars().filter(|c| !c.is_whitespace()).collect();
    let v: usize = compact
        .split("('outcomes','<u8',(")
        .nth(1)
        .and_then(|x| x.split(',').next())
        .and_then(|x| x.parse().ok())
        .ok_or("not a label shard (no outcomes field)")?;
    let want: String = descr(v).chars().filter(|c| !c.is_whitespace()).collect();
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
    Ok((n, v, start + hlen))
}

/// Read a shard written by [`to_npy`] (or by NumPy with the same record type).
pub fn from_npy(bytes: &[u8]) -> Result<Vec<Row>, String> {
    let (n, v, data) = parse_header(bytes)?;
    let rb = row_bytes_for(v);
    let body = &bytes[data..];
    if body.len() != n * rb {
        return Err(format!("{} data bytes for {n} records of {rb}", body.len()));
    }
    let mut rows = Vec::with_capacity(n);
    for rec in body.chunks(rb) {
        let u16_at = |i: usize| u16::from_le_bytes([rec[i], rec[i + 1]]);
        let u64_at = |i: usize| u64::from_le_bytes(rec[i..i + 8].try_into().expect("8 bytes"));
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
        let money = |at: usize| -> Result<MoneyChance, String> {
            MoneyChance::from_raw(core::array::from_fn(|t| u64_at(at + 8 * t)))
                .ok_or_else(|| "a money vector that does not sum to 2^32".to_string())
        };
        let value = money(o)?;
        let mut outcomes = Vec::with_capacity(v);
        for k in 0..v {
            outcomes.push(money(o + 32 + 2 + 1 + 8 + 32 * k)?);
        }
        rows.push(Row {
            features: f,
            value,
            source: rec[o + 32],
            trick: rec[o + 33],
            label: rec[o + 34],
            mask: u64_at(o + 35),
            outcomes,
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
        let outs = vec![
            MoneyChance::even(),
            MoneyChance::certain(3),
            MoneyChance::from_weights([1, 2, 3, 4]),
        ];
        let rows = vec![
            Row {
                features: f,
                value: MoneyChance::from_weights([1, 2, 3, 4]),
                source: SOURCE_RANDOM,
                trick: 4,
                label: 2,
                mask: 0b110,
                outcomes: outs.clone(),
            },
            Row {
                features: f,
                value: MoneyChance::certain(3),
                source: SOURCE_RULES,
                trick: 3,
                label: 1,
                mask: u64::MAX,
                outcomes: outs,
            },
        ];
        let b = to_npy(&rows, 3);
        assert_eq!(from_npy(&b).expect("reads"), rows);
        assert!(from_npy(&b[..b.len() - 1]).is_err());
    }
}
