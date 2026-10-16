//! Per-row loss weights of a label shard (docs/m3-hybrid/audit-loop.md §4 step 5: "rows inside
//! weak groups get a loss weight of 2"): a sidecar file `NAME.npy.rowweights` beside the shard
//! `NAME.npy`, read by the trainer (`python/nn/fortyone_nn/train.py`, `formats.load_row_weights`).
//!
//! **Format** (PLAN §12.7: readable by `numpy.load`, no Rust-only format): a `.npy` file holding
//! a one-dimensional array of `<u2` (little-endian u16), one per row of the shard in row order,
//! each the row's weight in units of 1/[`ROW_WEIGHT_ONE`] (256): 256 is a weight of 1, 512 a
//! weight of 2, 384 a weight of 1.5. Fixed point rather than a float, so that a weight has one
//! encoding (no −0, NaN or infinities to refuse, no rounding between writers), every value is
//! exact in the trainer's `float32`, and the writers stay in whole numbers like the rest of the
//! Rust side. The range is 1/256 to 65535/256 (about 256); 0 is refused (a weight-0 row would
//! only be a row to leave out, and a batch of them would have no loss). The header is format
//! version 1.0, padded so the data starts on a 64-byte boundary, as the shards' own.
//!
//! **Written before the shard is visible**: a trainer that watches a directory
//! (`train.py --watch-dir`) loads a shard's row weights when the shard appears under its final
//! name, so the sidecar must be complete before then: [`write`] puts it in place atomically
//! (another name, then a rename), and a writer of a streamed shard calls it before renaming the
//! shard itself into place (as `search labels` renames `NAME.npy.part`).
//!
//! In the loss each row's weight multiplies the shard's own weight (`NAME.npy.weight`, a number;
//! default 1) and every batch's loss is the weighted mean; a sidecar of all 256 trains exactly the
//! unweighted network.

use std::path::{Path, PathBuf};

/// The weight 1 in the sidecar's fixed point.
pub const ROW_WEIGHT_ONE: u16 = 256;

/// The sidecar's suffix, after the shard's own name.
pub const SUFFIX: &str = ".rowweights";

const MAGIC: &[u8; 6] = b"\x93NUMPY";
const DESCR: &str = "'<u2'";

/// The sidecar of the shard at `shard`: `<shard>.rowweights`.
pub fn path_for(shard: &Path) -> PathBuf {
    let mut s = shard.as_os_str().to_owned();
    s.push(SUFFIX);
    PathBuf::from(s)
}

/// A weight in the sidecar's fixed point, if it is a multiple of 1/256 from 1/256 to 65535/256.
pub fn fixed(weight: f64) -> Option<u16> {
    let x = weight * f64::from(ROW_WEIGHT_ONE);
    (x.fract() == 0.0 && (1.0..=f64::from(u16::MAX)).contains(&x)).then_some(x as u16)
}

/// The `.npy` bytes of `weights` (one per row).
///
/// # Errors
/// If a weight is 0.
pub fn to_npy(weights: &[u16]) -> Result<Vec<u8>, String> {
    if let Some(i) = weights.iter().position(|&w| w == 0) {
        return Err(format!(
            "row {i}: a weight of 0 (the smallest is 1, i.e. 1/256)"
        ));
    }
    let mut h = format!(
        "{{'descr': {DESCR}, 'fortran_order': False, 'shape': ({},), }}",
        weights.len()
    );
    let unpadded = 10 + h.len() + 1;
    h.push_str(&" ".repeat((64 - unpadded % 64) % 64));
    h.push('\n');
    let len = u16::try_from(h.len()).expect("header fits a u16");
    let mut out = Vec::with_capacity(10 + h.len() + 2 * weights.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    for w in weights {
        out.extend_from_slice(&w.to_le_bytes());
    }
    Ok(out)
}

/// The weights of a sidecar's bytes (as [`to_npy`] writes them, or `numpy.save` of a 1-D `<u2`
/// array).
///
/// # Errors
/// If the bytes are not such a file, or a weight is 0.
pub fn from_npy(bytes: &[u8]) -> Result<Vec<u16>, String> {
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
    if !squeeze.contains(&format!("'descr':{DESCR}")) || !squeeze.contains("'fortran_order':False")
    {
        return Err(format!(
            "row weights must be a 1-D '<u2' array, not {}",
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
        .map_err(|_| format!("row weights must be one-dimensional, not shape ({shape})"))?;
    let data = &bytes[start + hlen..];
    if data.len() != 2 * n {
        return Err(format!("{} data bytes for {n} weights", data.len()));
    }
    let w: Vec<u16> = data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&c| u16::from_le_bytes(c))
        .collect();
    if let Some(i) = w.iter().position(|&x| x == 0) {
        return Err(format!("row {i}: a weight of 0"));
    }
    Ok(w)
}

/// Writes the sidecar of the shard at `shard` whole and atomically (`<sidecar>.part`, then a
/// rename), so that a trainer watching the shard's directory never reads half of it. Call it
/// before the shard itself appears under its final name.
///
/// # Errors
/// As [`to_npy`], or on a file error.
pub fn write(shard: &Path, weights: &[u16]) -> Result<PathBuf, String> {
    let path = path_for(shard);
    let mut part = path.as_os_str().to_owned();
    part.push(".part");
    let part = PathBuf::from(part);
    std::fs::write(&part, to_npy(weights)?).map_err(|e| format!("{}: {e}", part.display()))?;
    std::fs::rename(&part, &path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}

/// The row weights of the shard at `shard`, if it has a sidecar.
///
/// # Errors
/// If the sidecar cannot be read or is not a row-weights file.
pub fn read(shard: &Path) -> Result<Option<Vec<u16>>, String> {
    let path = path_for(shard);
    match std::fs::read(&path) {
        Ok(b) => from_npy(&b)
            .map(Some)
            .map_err(|e| format!("{}: {e}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

/// The number of records of the label shard at `shard`, from its header alone (the records are
/// not read), checked against the file's length.
///
/// # Errors
/// If the file cannot be read, is not a label shard, or is not whole.
pub fn shard_rows(shard: &Path) -> Result<usize, String> {
    use std::io::Read;
    let name = shard.display();
    let f = std::fs::File::open(shard).map_err(|e| format!("{name}: {e}"))?;
    let len = f.metadata().map_err(|e| format!("{name}: {e}"))?.len();
    let mut head = Vec::new();
    f.take(1 << 16)
        .read_to_end(&mut head)
        .map_err(|e| format!("{name}: {e}"))?;
    let (n, v, data) = crate::shard::parse_header(&head).map_err(|e| format!("{name}: {e}"))?;
    let want = data as u64 + (n as u64) * (crate::shard::row_bytes_for(v) as u64);
    if len != want {
        return Err(format!(
            "{name}: {len} bytes, not the {want} of {n} records (not whole)"
        ));
    }
    Ok(n)
}

/// What a sidecar holds, in whole numbers: its rows, how many have a weight other than 1, the
/// sum of the weights (units of 1/256) and the smallest and largest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Summary {
    pub rows: usize,
    pub rows_not_1: usize,
    pub sum: u64,
    pub min: u16,
    pub max: u16,
}

/// The [`Summary`] of `weights` (min and max 0 when there are none).
pub fn summary(weights: &[u16]) -> Summary {
    Summary {
        rows: weights.len(),
        rows_not_1: weights.iter().filter(|&&w| w != ROW_WEIGHT_ONE).count(),
        sum: weights.iter().map(|&w| u64::from(w)).sum(),
        min: weights.iter().copied().min().unwrap_or(0),
        max: weights.iter().copied().max().unwrap_or(0),
    }
}

/// The weights of a shard of `rows` rows where the rows `chosen` (indices, in any order, repeats
/// allowed) weigh `weight` (fixed point) and the others 1.
///
/// # Errors
/// If an index is not a row, or the weight is 0.
pub fn chosen_rows(rows: usize, chosen: &[u64], weight: u16) -> Result<Vec<u16>, String> {
    if weight == 0 {
        return Err("a weight of 0 (the smallest is 1, i.e. 1/256)".into());
    }
    let mut w = vec![ROW_WEIGHT_ONE; rows];
    for &i in chosen {
        let slot = usize::try_from(i)
            .ok()
            .and_then(|i| w.get_mut(i))
            .ok_or_else(|| format!("row {i} of a shard of {rows} rows"))?;
        *slot = weight;
    }
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN_SHA: &str = "f28e3d2b6d7831072ef9c3a4e46913bd76e0ec1944c5054f70917560602933b2";

    #[test]
    fn round_trip_and_layout() {
        let w = [256u16, 512, 384, 1, u16::MAX];
        let b = to_npy(&w).expect("bytes");
        assert_eq!(&b[..8], b"\x93NUMPY\x01\x00");
        let hlen = usize::from(u16::from_le_bytes([b[8], b[9]]));
        assert_eq!((10 + hlen) % 64, 0, "data on a 64-byte boundary");
        assert_eq!(b.len(), 10 + hlen + 2 * w.len());
        assert_eq!(&b[10 + hlen..10 + hlen + 4], &[0, 1, 0, 2]);
        assert_eq!(from_npy(&b).expect("read"), w);
        assert!(to_npy(&[256, 0]).is_err());
        assert_eq!(
            from_npy(&to_npy(&[]).expect("empty")).expect("read"),
            Vec::<u16>::new()
        );
        // Golden bytes: a change of format is a change of this hash (numpy writes the same
        // bytes for `np.save(f, np.array([256, 512, 384, 1, 65535], dtype="<u2"))`, tested in
        // python/nn).
        assert_eq!(equity::sha256::sha256_hex(&b), GOLDEN_SHA);
    }

    #[test]
    fn fixed_point() {
        assert_eq!(fixed(1.0), Some(256));
        assert_eq!(fixed(2.0), Some(512));
        assert_eq!(fixed(1.5), Some(384));
        assert_eq!(fixed(1.0 / 256.0), Some(1));
        assert_eq!(fixed(0.0), None);
        assert_eq!(fixed(1.001), None, "not a multiple of 1/256");
        assert_eq!(fixed(256.0), None, "over 65535/256");
        assert_eq!(fixed(-1.0), None);
    }

    #[test]
    fn foreign_and_broken_files_are_refused() {
        let ok = to_npy(&[256, 512]).expect("bytes");
        let mut f4 = ok.clone();
        let at = f4.windows(5).position(|x| x == b"'<u2'").expect("descr");
        f4[at..at + 5].copy_from_slice(b"'<f4'");
        assert!(from_npy(&f4).is_err());
        assert!(from_npy(&ok[..ok.len() - 1]).is_err(), "truncated");
        assert!(from_npy(b"not npy").is_err());
        let mut zero = ok.clone();
        let n = zero.len();
        zero[n - 2..].copy_from_slice(&[0, 0]);
        assert!(from_npy(&zero).is_err());
    }

    #[test]
    fn written_atomically_beside_the_shard() {
        let d = std::env::temp_dir().join(format!("rowweights-{}", std::process::id()));
        std::fs::create_dir_all(&d).expect("mkdir");
        let shard = d.join("t3-harvest-s103.npy");
        assert_eq!(read(&shard).expect("none"), None);
        let p = write(&shard, &[256, 512, 512]).expect("written");
        assert_eq!(p, d.join("t3-harvest-s103.npy.rowweights"));
        assert!(!d.join("t3-harvest-s103.npy.rowweights.part").exists());
        assert_eq!(read(&shard).expect("read"), Some(vec![256, 512, 512]));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn rows_of_a_shard_from_its_header() {
        let d = std::env::temp_dir().join(format!("rowweights-rows-{}", std::process::id()));
        std::fs::create_dir_all(&d).expect("mkdir");
        let shard = d.join("s.npy");
        // The records are not read: zeros after a real header will do.
        let mut b = crate::shard::header(3, 14);
        b.resize(b.len() + 3 * crate::shard::row_bytes_for(14), 0);
        std::fs::write(&shard, &b).expect("write");
        assert_eq!(shard_rows(&shard), Ok(3));
        std::fs::write(&shard, &b[..b.len() - 1]).expect("write");
        assert!(shard_rows(&shard).expect_err("short").contains("not whole"));
        std::fs::write(&shard, to_npy(&[256]).expect("bytes")).expect("write");
        assert!(shard_rows(&shard).is_err(), "row weights are not a shard");
        assert!(shard_rows(&d.join("none.npy")).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn chosen_rows_and_summary() {
        let w = chosen_rows(5, &[3, 0, 3], 512).expect("weights");
        assert_eq!(w, [512, 256, 256, 512, 256]);
        assert_eq!(
            summary(&w),
            Summary {
                rows: 5,
                rows_not_1: 2,
                sum: 1792,
                min: 256,
                max: 512
            }
        );
        assert!(chosen_rows(5, &[5], 512).is_err(), "no row 5");
        assert!(chosen_rows(5, &[0], 0).is_err());
        assert_eq!(chosen_rows(2, &[], 512).expect("none chosen"), [256, 256]);
        assert_eq!(summary(&[]).rows_not_1, 0);
    }
}
