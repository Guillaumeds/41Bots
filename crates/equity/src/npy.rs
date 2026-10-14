//! A minimal `.npy` [NumPy's standard array file] writer and reader for `uint64` arrays (PLAN
//! §12.7, docs/M1-SPEC.md §1): format version 1.0, a hand-built header
//! `{'descr': '<u8', 'fortran_order': False, 'shape': (…), }` padded with spaces and a newline so
//! the data starts on a 64-byte boundary, then the values little-endian in C order. Python reads
//! it with `numpy.load`.

use core::fmt;

const MAGIC: &[u8; 6] = b"\x93NUMPY";

/// Why a `.npy` file could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NpyError(pub String);

impl fmt::Display for NpyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bad .npy file: {}", self.0)
    }
}

impl std::error::Error for NpyError {}

fn err<T>(msg: impl Into<String>) -> Result<T, NpyError> {
    Err(NpyError(msg.into()))
}

/// The bytes of a `.npy` file holding `data` (C order) with this shape, dtype `<u8`.
///
/// # Panics
/// If the shape's product is not `data.len()`.
pub fn to_npy_u64(shape: &[usize], data: &[u64]) -> Vec<u8> {
    assert_eq!(
        shape.iter().product::<usize>(),
        data.len(),
        "shape does not match the data"
    );
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let shape_text = if dims.len() == 1 {
        format!("({},)", dims[0])
    } else {
        format!("({})", dims.join(", "))
    };
    let mut header = format!("{{'descr': '<u8', 'fortran_order': False, 'shape': {shape_text}, }}");
    // magic (6) + version (2) + header length (2) + header, ending in '\n', to a multiple of 64.
    let unpadded = 10 + header.len() + 1;
    let pad = (64 - unpadded % 64) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    let header_len = u16::try_from(header.len()).expect("header fits in a u16");
    let mut out = Vec::with_capacity(10 + header.len() + 8 * data.len());
    out.extend_from_slice(MAGIC);
    out.push(1);
    out.push(0);
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for &v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    debug_assert_eq!((10 + usize::from(header_len)) % 64, 0);
    out
}

/// Read a `.npy` file holding little-endian `uint64` (`<u8`) values in C order (format
/// versions 1.0 to 3.0; anything else is refused). Returns the shape and the values in C order.
pub fn from_npy_u64(bytes: &[u8]) -> Result<(Vec<usize>, Vec<u64>), NpyError> {
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return err("missing the \\x93NUMPY magic");
    }
    let major = bytes[6];
    let (header_len, start) = match major {
        1 => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
        2 | 3 => {
            if bytes.len() < 12 {
                return err("truncated header");
            }
            let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
            (
                usize::try_from(len).map_err(|_| NpyError("header too long".into()))?,
                12,
            )
        }
        v => return err(format!("unsupported format version {v}")),
    };
    let data_start = start + header_len;
    if bytes.len() < data_start {
        return err("truncated header");
    }
    let header = core::str::from_utf8(&bytes[start..data_start])
        .map_err(|_| NpyError("header is not text".into()))?;
    let descr = dict_value(header, "descr")?;
    if descr != "'<u8'" {
        return err(format!("dtype {descr} is not '<u8' (little-endian uint64)"));
    }
    let order = dict_value(header, "fortran_order")?;
    if order != "False" {
        return err("fortran_order must be False");
    }
    let shape_text = dict_value(header, "shape")?;
    let inner = shape_text
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| NpyError(format!("bad shape {shape_text}")))?;
    let mut shape = Vec::new();
    for part in inner.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        shape.push(
            p.parse::<usize>()
                .map_err(|_| NpyError(format!("bad dimension `{p}`")))?,
        );
    }
    let n: usize = shape.iter().product();
    let body = &bytes[data_start..];
    if body.len() != n * 8 {
        return err(format!(
            "expected {} data bytes for shape {shape:?}, found {}",
            n * 8,
            body.len()
        ));
    }
    let data = body
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| u64::from_le_bytes(*c))
        .collect();
    Ok((shape, data))
}

/// The text of one value in the header dictionary (a quoted string, `True`/`False`, or a tuple).
fn dict_value<'a>(header: &'a str, key: &str) -> Result<&'a str, NpyError> {
    let quoted = format!("'{key}':");
    let pos = header
        .find(&quoted)
        .ok_or_else(|| NpyError(format!("header has no '{key}'")))?;
    let rest = header[pos + quoted.len()..].trim_start();
    let end = if rest.starts_with('(') {
        rest.find(')').map(|i| i + 1)
    } else if let Some(stripped) = rest.strip_prefix('\'') {
        stripped.find('\'').map(|i| i + 2)
    } else {
        rest.find([',', '}'])
    };
    let end = end.ok_or_else(|| NpyError(format!("unterminated value for '{key}'")))?;
    Ok(rest[..end].trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_exact_header() {
        let data: Vec<u64> = (0..24).map(|i| (i as u64) << 33 | 7).collect();
        let bytes = to_npy_u64(&[2, 3, 4], &data);
        assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
        let header_len = usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
        assert_eq!((10 + header_len) % 64, 0);
        let header = core::str::from_utf8(&bytes[10..10 + header_len]).expect("text");
        assert!(
            header.starts_with("{'descr': '<u8', 'fortran_order': False, 'shape': (2, 3, 4), }")
        );
        assert!(header.ends_with('\n'));
        // Little-endian values right after the header.
        assert_eq!(
            &bytes[10 + header_len..10 + header_len + 8],
            &7u64.to_le_bytes()
        );
        let (shape, back) = from_npy_u64(&bytes).expect("read");
        assert_eq!(shape, vec![2, 3, 4]);
        assert_eq!(back, data);
    }

    #[test]
    fn table_shape_header() {
        let bytes = to_npy_u64(
            &[16, 15, 15, 15, 15, 4],
            &vec![0; 16 * 15 * 15 * 15 * 15 * 4],
        );
        let (shape, data) = from_npy_u64(&bytes).expect("read");
        assert_eq!(shape, vec![16, 15, 15, 15, 15, 4]);
        assert_eq!(data.len(), 3_240_000);
    }

    #[test]
    fn one_dimension_uses_a_trailing_comma() {
        let bytes = to_npy_u64(&[3], &[1, 2, 3]);
        let text = String::from_utf8_lossy(&bytes[10..80]).to_string();
        assert!(text.contains("'shape': (3,)"), "{text}");
        assert_eq!(
            from_npy_u64(&bytes).expect("read"),
            (vec![3], vec![1, 2, 3])
        );
    }

    #[test]
    fn rejects_other_dtypes_and_truncation() {
        let mut bytes = to_npy_u64(&[2], &[1, 2]);
        assert!(from_npy_u64(&bytes[..bytes.len() - 1]).is_err());
        let pos = bytes.windows(4).position(|w| w == b"<u8'").expect("descr");
        bytes[pos + 2] = b'4';
        assert!(from_npy_u64(&bytes).is_err());
        assert!(from_npy_u64(b"not a numpy file").is_err());
    }
}
