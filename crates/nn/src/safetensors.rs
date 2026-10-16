//! A tiny reader and writer of safetensors files [a simple, safe, standard format for network
//! weights: an 8-byte little-endian header length, a JSON header naming each tensor's type,
//! shape and byte range, then the raw little-endian data] (PLAN §6.5, §12.7).
//!
//! Only what the networks need: `I32`, `I64` and `F32` tensors, and the string-to-string
//! `__metadata__` map. Written here instead of taking the `safetensors` crate so the inference
//! path has no dependency beyond `serde_json` and builds for every target (WASM included); the
//! format is checked against Python's reading of the same bytes in `python/nn`'s tests.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

/// A tensor's element type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    /// 32-bit signed integers.
    I32,
    /// 64-bit signed integers.
    I64,
    /// 32-bit floats (training-side files only; inference never reads them).
    F32,
}

impl Dtype {
    fn name(self) -> &'static str {
        match self {
            Dtype::I32 => "I32",
            Dtype::I64 => "I64",
            Dtype::F32 => "F32",
        }
    }

    fn size(self) -> usize {
        match self {
            Dtype::I32 | Dtype::F32 => 4,
            Dtype::I64 => 8,
        }
    }

    fn parse(s: &str) -> Option<Dtype> {
        match s {
            "I32" => Some(Dtype::I32),
            "I64" => Some(Dtype::I64),
            "F32" => Some(Dtype::F32),
            _ => None,
        }
    }
}

/// One tensor: type, shape, raw little-endian bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    /// Element type.
    pub dtype: Dtype,
    /// Shape (C order).
    pub shape: Vec<usize>,
    /// The data, little-endian.
    pub data: Vec<u8>,
}

impl Tensor {
    /// An `I32` tensor.
    pub fn i32(shape: Vec<usize>, v: &[i32]) -> Tensor {
        Tensor {
            dtype: Dtype::I32,
            shape,
            data: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }

    /// An `I64` tensor.
    pub fn i64(shape: Vec<usize>, v: &[i64]) -> Tensor {
        Tensor {
            dtype: Dtype::I64,
            shape,
            data: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }

    /// The values of an `I32` tensor.
    pub fn as_i32(&self) -> Result<Vec<i32>, String> {
        if self.dtype != Dtype::I32 {
            return Err(format!("expected I32, found {}", self.dtype.name()));
        }
        Ok(self
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b))
            .collect())
    }

    /// The values of an `I64` tensor.
    pub fn as_i64(&self) -> Result<Vec<i64>, String> {
        if self.dtype != Dtype::I64 {
            return Err(format!("expected I64, found {}", self.dtype.name()));
        }
        Ok(self
            .data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b))
            .collect())
    }
}

/// A safetensors file: named tensors and the metadata map.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct File {
    /// Tensors by name.
    pub tensors: BTreeMap<String, Tensor>,
    /// The `__metadata__` map.
    pub meta: BTreeMap<String, String>,
}

impl File {
    /// A tensor by name.
    pub fn get(&self, name: &str) -> Result<&Tensor, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("no tensor `{name}`"))
    }

    /// A metadata entry by name.
    pub fn meta(&self, key: &str) -> Result<&str, String> {
        self.meta
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| format!("no metadata `{key}`"))
    }
}

/// Read a safetensors file.
pub fn read(bytes: &[u8]) -> Result<File, String> {
    if bytes.len() < 8 {
        return Err("too short for a safetensors header".into());
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes"));
    let n = usize::try_from(n).map_err(|_| "header length".to_string())?;
    let end = 8usize
        .checked_add(n)
        .filter(|&e| e <= bytes.len())
        .ok_or("header runs past the end of the file")?;
    let header: Value =
        serde_json::from_slice(&bytes[8..end]).map_err(|e| format!("header: {e}"))?;
    let obj = header.as_object().ok_or("header is not a JSON object")?;
    let data = &bytes[end..];
    let mut out = File::default();
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for (name, v) in obj {
        if name == "__metadata__" {
            for (k, x) in v.as_object().ok_or("metadata is not an object")? {
                let s = x.as_str().ok_or("metadata values are strings")?;
                out.meta.insert(k.clone(), s.to_string());
            }
            continue;
        }
        let t = v
            .as_object()
            .ok_or_else(|| format!("{name}: not an object"))?;
        let dtype = t
            .get("dtype")
            .and_then(Value::as_str)
            .and_then(Dtype::parse)
            .ok_or_else(|| format!("{name}: unsupported dtype"))?;
        let shape: Vec<usize> = t
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{name}: no shape"))?
            .iter()
            .map(|d| d.as_u64().map(|x| x as usize))
            .collect::<Option<_>>()
            .ok_or_else(|| format!("{name}: bad shape"))?;
        let offs = t
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|a| a.len() == 2)
            .ok_or_else(|| format!("{name}: no data_offsets"))?;
        let (a, b) = (
            offs[0].as_u64().ok_or("offset")? as usize,
            offs[1].as_u64().ok_or("offset")? as usize,
        );
        let len = shape.iter().product::<usize>() * dtype.size();
        if a > b || b > data.len() || b - a != len {
            return Err(format!(
                "{name}: data range {a}..{b} does not fit {shape:?}"
            ));
        }
        ranges.push((a, b));
        out.tensors.insert(
            name.clone(),
            Tensor {
                dtype,
                shape,
                data: data[a..b].to_vec(),
            },
        );
    }
    ranges.sort_unstable();
    for w in ranges.windows(2) {
        if w[0].1 > w[1].0 {
            return Err("tensors overlap".into());
        }
    }
    Ok(out)
}

/// Write a safetensors file: tensors in name order, the header padded with spaces to a multiple
/// of 8 bytes (as the reference implementation does).
pub fn write(f: &File) -> Vec<u8> {
    let mut header = Map::new();
    if !f.meta.is_empty() {
        let m: Map<String, Value> = f
            .meta
            .iter()
            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect();
        header.insert("__metadata__".into(), Value::Object(m));
    }
    let mut off = 0usize;
    for (name, t) in &f.tensors {
        let mut o = Map::new();
        o.insert("dtype".into(), Value::String(t.dtype.name().into()));
        o.insert("shape".into(), Value::from(t.shape.clone()));
        o.insert(
            "data_offsets".into(),
            Value::from(vec![off, off + t.data.len()]),
        );
        off += t.data.len();
        header.insert(name.clone(), Value::Object(o));
    }
    let mut h = Value::Object(header).to_string().into_bytes();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + h.len() + off);
    out.extend_from_slice(&(h.len() as u64).to_le_bytes());
    out.extend_from_slice(&h);
    for t in f.tensors.values() {
        out.extend_from_slice(&t.data);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut f = File::default();
        f.tensors.insert(
            "w0".into(),
            Tensor::i32(vec![2, 3], &[1, -2, 3, i32::MIN, 5, i32::MAX]),
        );
        f.tensors
            .insert("b0".into(), Tensor::i64(vec![3], &[i64::MIN, 0, i64::MAX]));
        f.meta.insert("format".into(), "x".into());
        let bytes = write(&f);
        assert_eq!(bytes.len() % 8, 0);
        let g = read(&bytes).expect("reads");
        assert_eq!(f, g);
        assert_eq!(g.get("w0").unwrap().as_i32().unwrap()[3], i32::MIN);
        assert_eq!(g.get("b0").unwrap().as_i64().unwrap()[2], i64::MAX);
        assert!(g.get("b0").unwrap().as_i32().is_err());
    }

    #[test]
    fn refuses_bad_files() {
        assert!(read(&[1, 2, 3]).is_err());
        let mut bytes = write(&File::default());
        bytes[0] = 200;
        assert!(read(&bytes).is_err());
        // A data range that runs past the end.
        let mut f = File::default();
        f.tensors
            .insert("w".into(), Tensor::i32(vec![4], &[1, 2, 3, 4]));
        let mut b = write(&f);
        b.truncate(b.len() - 4);
        assert!(read(&b).is_err());
    }
}
