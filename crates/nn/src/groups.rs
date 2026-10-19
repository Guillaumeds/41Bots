//! Sibling groups of a label shard (owner, 2026-10-19: "fix E", decision-focused training): a
//! sidecar file `NAME.npy.groups` beside a shard `NAME.npy` whose rows come in groups of frontier
//! positions that one choice point separates (`search labels --source siblings`), read by the
//! trainer's pairwise ranking loss (`python/nn/fortyone_nn/train.py --pair-weight`,
//! `formats.load_groups`).
//!
//! **Format** (PLAN §12.7: readable by `numpy.load`, no Rust-only format): a `.npy` file holding a
//! one-dimensional array of packed records `[('group', '<u4'), ('mover', '|u1')]`, one per row of
//! the shard in row order:
//! - `group`: the row's group, numbered 0, 1, 2, … in row order; the rows of a group are
//!   consecutive, so the numbers never decrease and step by at most 1.
//! - `mover`: the team that chose at the group's choice point (the "mover", whose money chance
//!   the group's positions are compared by), as an index into **this row's** value order (the
//!   encoder's rotated team order: the same team can have another index in a sibling row whose
//!   trick-3 leader differs), 0–3.
//!
//! The header is format version 1.0, padded so the data starts on a 64-byte boundary, as the
//! shards' own. The shard itself is an ordinary label shard (its rows' `source` is
//! [`crate::shard::SOURCE_SIBLINGS`]); a reader that ignores the sidecar trains on its rows as on
//! any others, so old readers stay right and old shards (no sidecar) read as before.
//!
//! **Written before the shard is visible**, as the row weights ([`crate::rowweights::write`]):
//! atomically (another name, then a rename), before the shard is renamed into place.

use std::path::{Path, PathBuf};

/// The sidecar's suffix, after the shard's own name.
pub const SUFFIX: &str = ".groups";

const MAGIC: &[u8; 6] = b"\x93NUMPY";
const DESCR: &str = "[('group', '<u4'), ('mover', '|u1')]";

/// Bytes a record.
pub const RECORD_BYTES: usize = 5;

/// One row's group and mover.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupRow {
    /// The group, numbered in row order from 0.
    pub group: u32,
    /// The mover's team as an index into the row's (rotated) value order.
    pub mover: u8,
}

/// The sidecar of the shard at `shard`: `<shard>.groups`.
pub fn path_for(shard: &Path) -> PathBuf {
    let mut s = shard.as_os_str().to_owned();
    s.push(SUFFIX);
    PathBuf::from(s)
}

/// Whether `rows` are a valid group list: groups numbered from 0 in row order (each row's group
/// equal to the row before's or one more), movers 0–3.
///
/// # Errors
/// The first row that breaks a rule.
pub fn check(rows: &[GroupRow]) -> Result<(), String> {
    let mut prev: Option<u32> = None;
    for (i, r) in rows.iter().enumerate() {
        if r.mover > 3 {
            return Err(format!("row {i}: mover {} (a team index is 0-3)", r.mover));
        }
        let ok = match prev {
            None => r.group == 0,
            Some(p) => r.group == p || Some(r.group) == p.checked_add(1),
        };
        if !ok {
            return Err(format!(
                "row {i}: group {} after {prev:?} (groups are numbered 0, 1, 2, … in row order)",
                r.group
            ));
        }
        prev = Some(r.group);
    }
    Ok(())
}

/// The `.npy` bytes of `rows` (one per shard row).
///
/// # Errors
/// As [`check`].
pub fn to_npy(rows: &[GroupRow]) -> Result<Vec<u8>, String> {
    check(rows)?;
    let mut h = format!(
        "{{'descr': {DESCR}, 'fortran_order': False, 'shape': ({},), }}",
        rows.len()
    );
    let unpadded = 10 + h.len() + 1;
    h.push_str(&" ".repeat((64 - unpadded % 64) % 64));
    h.push('\n');
    let len = u16::try_from(h.len()).expect("header fits a u16");
    let mut out = Vec::with_capacity(10 + h.len() + RECORD_BYTES * rows.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    for r in rows {
        out.extend_from_slice(&r.group.to_le_bytes());
        out.push(r.mover);
    }
    Ok(out)
}

/// The rows of a sidecar's bytes (as [`to_npy`] writes them, or `numpy.save` of the same record
/// type).
///
/// # Errors
/// If the bytes are not such a file, or break a rule of [`check`].
pub fn from_npy(bytes: &[u8]) -> Result<Vec<GroupRow>, String> {
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err("not a .npy file".into());
    }
    let (hlen, start) = match bytes[6] {
        1 => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
        2 | 3 if bytes.len() >= 12 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        v => return Err(format!(".npy format version {v}")),
    };
    let head = bytes
        .get(start..start + hlen)
        .and_then(|h| std::str::from_utf8(h).ok())
        .ok_or("a truncated .npy header")?;
    let squeeze: String = head.chars().filter(|c| !c.is_whitespace()).collect();
    let want: String = DESCR.chars().filter(|c| !c.is_whitespace()).collect();
    if !squeeze.contains(&format!("'descr':{want}")) || !squeeze.contains("'fortran_order':False") {
        return Err(format!(
            "groups must be a 1-D {DESCR} array, not {}",
            head.trim()
        ));
    }
    let shape = squeeze
        .split("'shape':(")
        .nth(1)
        .and_then(|r| r.split(')').next())
        .ok_or("no shape in the .npy header")?;
    let n: usize = shape
        .trim_end_matches(',')
        .parse()
        .map_err(|_| format!("groups must be one-dimensional, not shape ({shape})"))?;
    let data = &bytes[start + hlen..];
    if data.len() != RECORD_BYTES * n {
        return Err(format!("{} data bytes for {n} group rows", data.len()));
    }
    let rows: Vec<GroupRow> = data
        .as_chunks::<RECORD_BYTES>()
        .0
        .iter()
        .map(|c| GroupRow {
            group: u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
            mover: c[4],
        })
        .collect();
    check(&rows)?;
    Ok(rows)
}

/// Writes the sidecar of the shard at `shard` whole and atomically (`<sidecar>.part`, then a
/// rename). Call it before the shard itself appears under its final name.
///
/// # Errors
/// As [`to_npy`], or on a file error.
pub fn write(shard: &Path, rows: &[GroupRow]) -> Result<PathBuf, String> {
    let path = path_for(shard);
    let mut part = path.as_os_str().to_owned();
    part.push(".part");
    let part = PathBuf::from(part);
    std::fs::write(&part, to_npy(rows)?).map_err(|e| format!("{}: {e}", part.display()))?;
    std::fs::rename(&part, &path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}

/// The group rows of the shard at `shard`, if it has a sidecar.
///
/// # Errors
/// If the sidecar cannot be read or is not a groups file.
pub fn read(shard: &Path) -> Result<Option<Vec<GroupRow>>, String> {
    let path = path_for(shard);
    match std::fs::read(&path) {
        Ok(b) => from_npy(&b)
            .map(Some)
            .map_err(|e| format!("{}: {e}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

/// What a sidecar holds: rows, groups, and the number of groups of each size (index = size, the
/// last entry counting every size at or above it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    pub rows: usize,
    pub groups: usize,
    pub by_size: Vec<u64>,
}

/// Sizes counted one by one in [`Summary::by_size`]; larger groups count in the last entry.
pub const SIZES: usize = 9;

/// The sidecar's [`Summary`].
pub fn summary(rows: &[GroupRow]) -> Summary {
    let mut by_size = vec![0u64; SIZES + 1];
    let mut i = 0;
    let mut groups = 0;
    while i < rows.len() {
        let g = rows[i].group;
        let mut j = i;
        while j < rows.len() && rows[j].group == g {
            j += 1;
        }
        by_size[(j - i).min(SIZES)] += 1;
        groups += 1;
        i = j;
    }
    Summary {
        rows: rows.len(),
        groups,
        by_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(v: &[(u32, u8)]) -> Vec<GroupRow> {
        v.iter()
            .map(|&(group, mover)| GroupRow { group, mover })
            .collect()
    }

    #[test]
    fn round_trip_and_layout() {
        let r = rows(&[(0, 1), (0, 3), (1, 0), (1, 0), (1, 2), (2, 0)]);
        let b = to_npy(&r).expect("valid");
        let start = 10 + usize::from(u16::from_le_bytes([b[8], b[9]]));
        assert_eq!(start % 64, 0, "data on a 64-byte boundary");
        assert_eq!(b.len(), start + 5 * r.len());
        assert_eq!(&b[start..start + 5], &[0, 0, 0, 0, 1]);
        assert_eq!(from_npy(&b).expect("reads"), r);
        let s = summary(&r);
        assert_eq!((s.rows, s.groups), (6, 3));
        assert_eq!(s.by_size[1..4], [1, 1, 1]);
    }

    #[test]
    fn broken_lists_are_refused() {
        assert!(to_npy(&rows(&[(1, 0)])).is_err(), "must start at 0");
        assert!(to_npy(&rows(&[(0, 0), (2, 0)])).is_err(), "no gaps");
        assert!(
            to_npy(&rows(&[(0, 0), (1, 0), (0, 0)])).is_err(),
            "never back"
        );
        assert!(to_npy(&rows(&[(0, 4)])).is_err(), "teams 0-3");
        assert!(to_npy(&[]).is_ok());
        let w = crate::rowweights::to_npy(&[256, 256]).expect("weights");
        assert!(
            from_npy(&w).is_err(),
            "a row-weights file is not a groups file"
        );
        let mut b = to_npy(&rows(&[(0, 0), (0, 1)])).expect("valid");
        b.pop();
        assert!(from_npy(&b).is_err(), "truncated");
    }

    #[test]
    fn written_atomically_beside_the_shard() {
        let dir = std::env::temp_dir().join(format!("groups-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let shard = dir.join("x.npy");
        assert_eq!(read(&shard).expect("absent is fine"), None);
        let r = rows(&[(0, 2), (0, 2)]);
        let p = write(&shard, &r).expect("written");
        assert_eq!(p, dir.join("x.npy.groups"));
        assert!(!dir.join("x.npy.groups.part").exists());
        assert_eq!(read(&shard).expect("reads"), Some(r));
        std::fs::remove_dir_all(&dir).expect("clean");
    }
}
