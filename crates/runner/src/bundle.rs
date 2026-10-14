//! Output bundles: a deterministic `.tar.zst` of one chunk's output directory, and SHA-256.
//!
//! The tar is plain ustar written here (sorted relative paths with `/`, mode 0644, owner 0,
//! mtime 0, regular files only), compressed by zstd level 3 in one thread, so the same files give
//! the same bundle bytes on every machine and operating system (for the pinned zstd version).

use std::fs;
use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

/// Lowercase hex SHA-256 of bytes.
pub fn sha256_hex(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// SHA-256 and byte count of everything a reader yields (streams, so large files are fine).
pub fn sha256_reader(mut r: impl Read) -> Result<(String, u64), String> {
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    let mut n = 0u64;
    loop {
        let k = r.read(&mut buf).map_err(|e| e.to_string())?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        n += k as u64;
    }
    Ok((hex(&h.finalize()), n))
}

/// SHA-256 and size of a file.
pub fn sha256_file(p: &Path) -> Result<(String, u64), String> {
    sha256_reader(fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?)
}

/// Every regular file under `dir`, as sorted relative `/` paths.
pub fn list_files(dir: &Path) -> Result<Vec<String>, String> {
    fn walk(base: &Path, d: &Path, out: &mut Vec<String>) -> Result<(), String> {
        for e in fs::read_dir(d).map_err(|e| format!("{}: {e}", d.display()))? {
            let e = e.map_err(|e| e.to_string())?;
            let p = e.path();
            let t = e.file_type().map_err(|e| e.to_string())?;
            if t.is_dir() {
                walk(base, &p, out)?;
            } else if t.is_file() {
                let rel = p.strip_prefix(base).map_err(|e| e.to_string())?;
                let parts: Vec<String> = rel
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy().into_owned())
                    .collect();
                out.push(parts.join("/"));
            }
        }
        Ok(())
    }
    let mut v = Vec::new();
    walk(dir, dir, &mut v)?;
    v.sort();
    Ok(v)
}

fn octal(field: &mut [u8], v: u64) {
    let s = format!("{v:0w$o}", w = field.len() - 1);
    field[..s.len()].copy_from_slice(s.as_bytes());
    field[field.len() - 1] = 0;
}

fn header(name: &str, size: u64) -> Result<[u8; 512], String> {
    let mut h = [0u8; 512];
    let nb = name.as_bytes();
    if nb.len() <= 100 {
        h[..nb.len()].copy_from_slice(nb);
    } else {
        // ustar prefix/name split at a '/'.
        let cut = (0..nb.len())
            .rev()
            .find(|&i| nb[i] == b'/' && i <= 155 && nb.len() - i - 1 <= 100)
            .ok_or_else(|| format!("path too long for ustar: {name}"))?;
        h[..nb.len() - cut - 1].copy_from_slice(&nb[cut + 1..]);
        h[345..345 + cut].copy_from_slice(&nb[..cut]);
    }
    octal(&mut h[100..108], 0o644);
    octal(&mut h[108..116], 0);
    octal(&mut h[116..124], 0);
    octal(&mut h[124..136], size);
    octal(&mut h[136..148], 0);
    h[156] = b'0';
    h[257..263].copy_from_slice(b"ustar\0");
    h[263..265].copy_from_slice(b"00");
    h[148..156].copy_from_slice(b"        ");
    let sum: u64 = h.iter().map(|&b| u64::from(b)).sum();
    let s = format!("{sum:06o}\0 ");
    h[148..156].copy_from_slice(s.as_bytes());
    Ok(h)
}

/// The uncompressed tar of `dir`.
pub fn tar_dir(dir: &Path) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for rel in list_files(dir)? {
        let data = fs::read(dir.join(&rel)).map_err(|e| format!("{rel}: {e}"))?;
        out.extend_from_slice(&header(&rel, data.len() as u64)?);
        out.extend_from_slice(&data);
        out.resize(out.len().div_ceil(512) * 512, 0);
    }
    out.resize(out.len() + 1024, 0);
    Ok(out)
}

/// Writes `dir` as a `.tar.zst` to `dest`; returns (sha256, bytes) of the bundle.
pub fn bundle_dir(dir: &Path, dest: &Path) -> Result<(String, u64), String> {
    let tar = tar_dir(dir)?;
    let z = zstd::bulk::compress(&tar, 3).map_err(|e| format!("zstd: {e}"))?;
    if let Some(d) = dest.parent() {
        fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    fs::write(dest, &z).map_err(|e| format!("{}: {e}", dest.display()))?;
    Ok((sha256_hex(&z), z.len() as u64))
}

/// Reads one file back out of a bundle (tests).
#[cfg(test)]
pub fn read_member(bundle: &[u8], name: &str) -> Result<Option<Vec<u8>>, String> {
    let tar = zstd::stream::decode_all(bundle).map_err(|e| format!("zstd: {e}"))?;
    let mut at = 0usize;
    while at + 512 <= tar.len() && tar[at] != 0 {
        let h = &tar[at..at + 512];
        let field = |r: std::ops::Range<usize>| {
            String::from_utf8_lossy(&h[r])
                .trim_end_matches('\0')
                .to_string()
        };
        let mut n = field(0..100);
        let prefix = field(345..500);
        if !prefix.is_empty() {
            n = format!("{prefix}/{n}");
        }
        let size =
            u64::from_str_radix(field(124..136).trim(), 8).map_err(|e| e.to_string())? as usize;
        let start = at + 512;
        if n == name {
            return Ok(tar.get(start..start + size).map(<[u8]>::to_vec));
        }
        at = start + size.div_ceil(512) * 512;
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn bundle_is_deterministic_and_readable() {
        let base = std::env::temp_dir().join(format!("runner-bundle-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let d = base.join("src");
        fs::create_dir_all(d.join("sub")).unwrap();
        fs::write(d.join("b.txt"), b"bee").unwrap();
        fs::write(d.join("sub").join("a.json"), b"{}\n").unwrap();
        let (s1, n1) = bundle_dir(&d, &base.join("1.tar.zst")).unwrap();
        let (s2, n2) = bundle_dir(&d, &base.join("2.tar.zst")).unwrap();
        assert_eq!((s1, n1), (s2, n2));
        let z = fs::read(base.join("1.tar.zst")).unwrap();
        assert_eq!(
            read_member(&z, "sub/a.json").unwrap(),
            Some(b"{}\n".to_vec())
        );
        assert_eq!(read_member(&z, "b.txt").unwrap(), Some(b"bee".to_vec()));
        assert_eq!(read_member(&z, "nope").unwrap(), None);
        let _ = fs::remove_dir_all(&base);
    }
}
