//! The module's exports (docs/WASM-API.md §2): bytes in and out through Rust-owned buffers.
//!
//! JavaScript asks for an input buffer of `len` bytes ([`inbox`], which returns its address in
//! the module's memory), writes the bytes there through the exported `memory`, then calls an
//! operation with `len`; the operation leaves its output in a Rust-owned output buffer whose
//! address and length JavaScript reads with [`outbox`] and [`outbox_len`]. Rust only ever reads
//! and writes the two vectors it owns, in ordinary safe code; the only exception to the crate's
//! lint is the `no_mangle` attribute every export needs (allowed item by item).
//!
//! Natively the same functions exist (the tests call them); there an address is meaningless and
//! only the lengths and the buffers' contents are used.

use std::sync::{Mutex, MutexGuard};

use crate::api::{API_VERSION, Engine, merge as merge_json, refusal_json};
use crate::appview::Refusal;

static INBOX: Mutex<Vec<u8>> = Mutex::new(Vec::new());
static OUTBOX: Mutex<Vec<u8>> = Mutex::new(Vec::new());
static ENGINE: Mutex<Option<Engine>> = Mutex::new(None);
/// The SHA-256 (hex) the next file loaded must have, if any ([`expect_sha256`]).
static EXPECTED: Mutex<Option<String>> = Mutex::new(None);

fn lock<T>(m: &'static Mutex<T>) -> MutexGuard<'static, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The first `len` bytes of the input buffer, taken out of it (the buffer is left empty, so a
/// large file does not stay in memory twice).
fn take_input(len: u32) -> Vec<u8> {
    let mut v = std::mem::take(&mut *lock(&INBOX));
    v.truncate(len as usize);
    v
}

/// Leave `bytes` in the output buffer; returns their length.
fn put_output(bytes: Vec<u8>) -> u32 {
    let n = bytes.len() as u32;
    *lock(&OUTBOX) = bytes;
    n
}

/// The output buffer's contents (tests and the native binary).
pub fn output() -> Vec<u8> {
    lock(&OUTBOX).clone()
}

/// The input buffer, for native callers: what JavaScript does through the memory.
pub fn set_input(bytes: &[u8]) -> u32 {
    let n = bytes.len() as u32;
    let mut v = lock(&INBOX);
    v.clear();
    v.extend_from_slice(bytes);
    n
}

fn engine_mut() -> MutexGuard<'static, Option<Engine>> {
    let mut e = lock(&ENGINE);
    if e.is_none() {
        *e = Some(Engine::new());
    }
    e
}

/// The module's memory in bytes (WebAssembly only).
fn memory_bytes() -> Option<u64> {
    #[cfg(target_family = "wasm")]
    {
        Some(core::arch::wasm32::memory_size(0) as u64 * 65_536)
    }
    #[cfg(not(target_family = "wasm"))]
    {
        None
    }
}

/// The file in the input buffer, checked against the expected SHA-256 if one was given (the
/// expectation holds for one load only).
fn take_file(len: u32) -> Result<Vec<u8>, Refusal> {
    let bytes = take_input(len);
    if let Some(want) = lock(&EXPECTED).take() {
        let got = equity::sha256::sha256_hex(&bytes);
        if got != want {
            return Err(Refusal::new(
                "sha256_mismatch",
                format!("the file's SHA-256 is {got}, not {want}"),
            ));
        }
    }
    Ok(bytes)
}

fn status(r: Result<String, Refusal>) -> u32 {
    match r {
        Ok(json) => {
            put_output(json.into_bytes());
            0
        }
        Err(e) => {
            put_output(refusal_json(&e).into_bytes());
            1
        }
    }
}

/// The API version (1).
#[allow(unsafe_code)] // An export needs the `no_mangle` attribute; nothing else here is unsafe.
#[unsafe(no_mangle)]
pub extern "C" fn api_version() -> u32 {
    API_VERSION
}

/// Make the input buffer `len` bytes long (zeros) and return its address in the module's memory.
#[allow(unsafe_code)] // An export needs the `no_mangle` attribute; nothing else here is unsafe.
#[unsafe(no_mangle)]
pub extern "C" fn inbox(len: u32) -> u32 {
    let mut v = lock(&INBOX);
    v.clear();
    v.shrink_to(len as usize);
    v.resize(len as usize, 0);
    v.as_mut_ptr() as usize as u32
}

/// The output buffer's address.
#[allow(unsafe_code)] // An export needs the `no_mangle` attribute; nothing else here is unsafe.
#[unsafe(no_mangle)]
pub extern "C" fn outbox() -> u32 {
    lock(&OUTBOX).as_ptr() as usize as u32
}

/// The output buffer's length.
#[allow(unsafe_code)] // An export needs the `no_mangle` attribute; nothing else here is unsafe.
#[unsafe(no_mangle)]
pub extern "C" fn outbox_len() -> u32 {
    lock(&OUTBOX).len() as u32
}

/// The SHA-256 (64 lowercase hex digits, in the input buffer) the next file loaded must have:
/// a file that differs is refused and nothing is loaded. 0 (taken) or 1 (not a SHA-256: the
/// output holds a refusal).
#[allow(unsafe_code)] // An export needs the `no_mangle` attribute; nothing else here is unsafe.
#[unsafe(no_mangle)]
pub extern "C" fn expect_sha256(len: u32) -> u32 {
    let bytes = take_input(len);
    let ok = bytes.len() == 64
        && bytes
            .iter()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b));
    if !ok {
        return status(Err(Refusal::new(
            "bad_request",
            "a SHA-256 is 64 lowercase hex digits",
        )));
    }
    *lock(&EXPECTED) = Some(String::from_utf8(bytes).expect("ASCII"));
    status(Ok("{}".to_string()))
}

/// Load the equity table from the input buffer: 0 (the output holds `{"sha256":…}`) or 1 (the
/// output holds a refusal).
#[allow(unsafe_code)] // An export needs the `no_mangle` attribute; nothing else here is unsafe.
#[unsafe(no_mangle)]
pub extern "C" fn load_equity(len: u32) -> u32 {
    let r = take_file(len).and_then(|bytes| {
        engine_mut()
            .as_mut()
            .expect("made above")
            .load_equity(&bytes)
            .map(|sha| format!("{{\"sha256\":\"{sha}\",\"bytes\":{}}}", bytes.len()))
    });
    status(r)
}

/// Load an oracle network from the input buffer: 0 (the output holds
/// `{"sha256":…,"trick":…,"bytes":…}`) or 1 (the output holds a refusal).
#[allow(unsafe_code)] // An export needs the `no_mangle` attribute; nothing else here is unsafe.
#[unsafe(no_mangle)]
pub extern "C" fn load_net(len: u32) -> u32 {
    let r = take_file(len).and_then(|bytes| {
        engine_mut()
            .as_mut()
            .expect("made above")
            .load_net(&bytes)
            .map(|(sha, trick)| {
                format!(
                    "{{\"sha256\":\"{sha}\",\"trick\":{trick},\"bytes\":{}}}",
                    bytes.len()
                )
            })
    });
    status(r)
}

/// Load the student network from the input buffer: 0 (the output holds `{"sha256":…,"bytes":…}`)
/// or 1 (the output holds a refusal).
#[allow(unsafe_code)] // An export needs the `no_mangle` attribute; nothing else here is unsafe.
#[unsafe(no_mangle)]
pub extern "C" fn load_student(len: u32) -> u32 {
    let r = take_file(len).and_then(|bytes| {
        engine_mut()
            .as_mut()
            .expect("made above")
            .load_student(&bytes)
            .map(|sha| format!("{{\"sha256\":\"{sha}\",\"bytes\":{}}}", bytes.len()))
    });
    status(r)
}

/// Load a student network for one trick (1 or 2) from the input buffer: 0 (the output holds
/// `{"sha256":…,"trick":…,"bytes":…}`) or 1 (the output holds a refusal). It serves
/// `"method": "student"` at that trick, ahead of the default network of `load_student`.
#[allow(unsafe_code)] // An export needs the `no_mangle` attribute; nothing else here is unsafe.
#[unsafe(no_mangle)]
pub extern "C" fn load_student_trick(len: u32, trick: u32) -> u32 {
    let r = take_file(len).and_then(|bytes| {
        let t = u8::try_from(trick).unwrap_or(0);
        engine_mut()
            .as_mut()
            .expect("made above")
            .load_student_for(&bytes, t)
            .map(|sha| {
                format!(
                    "{{\"sha256\":\"{sha}\",\"trick\":{t},\"bytes\":{}}}",
                    bytes.len()
                )
            })
    });
    status(r)
}

/// Value the request in the input buffer; the result (or refusal) JSON goes to the output.
/// Returns its length.
#[allow(unsafe_code)] // An export needs the `no_mangle` attribute; nothing else here is unsafe.
#[unsafe(no_mangle)]
pub extern "C" fn evaluate(len: u32) -> u32 {
    let bytes = take_input(len);
    let out = match std::str::from_utf8(&bytes) {
        Ok(text) => engine_mut().as_ref().expect("made above").evaluate(text),
        Err(_) => refusal_json(&Refusal::new("bad_request", "the request is not UTF-8")),
    };
    put_output(out.into_bytes())
}

/// Merge the JSON array of results in the input buffer; the merged result (or refusal) goes to
/// the output. Returns its length.
#[allow(unsafe_code)] // An export needs the `no_mangle` attribute; nothing else here is unsafe.
#[unsafe(no_mangle)]
pub extern "C" fn merge(len: u32) -> u32 {
    let bytes = take_input(len);
    let out = match std::str::from_utf8(&bytes) {
        Ok(text) => merge_json(text),
        Err(_) => refusal_json(&Refusal::new("bad_request", "the results are not UTF-8")),
    };
    put_output(out.into_bytes())
}

/// The engine's version, the loaded files and the memory in use, as JSON in the output. Returns
/// its length.
#[allow(unsafe_code)] // An export needs the `no_mangle` attribute; nothing else here is unsafe.
#[unsafe(no_mangle)]
pub extern "C" fn info() -> u32 {
    let s = engine_mut()
        .as_ref()
        .expect("made above")
        .info(memory_bytes());
    put_output(s.into_bytes())
}
