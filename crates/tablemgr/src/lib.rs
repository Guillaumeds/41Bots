//! The 41 table (docs/M1-SPEC.md §3, PLAN §6.2).
//!
//! The table is the only code that holds an [`engine::game::Match`]. It deals, asks the seat to
//! act for each decision, refuses any action that is not legal, and writes a [`MatchRecord`] of
//! every action and the money. Each seat agent receives nothing but its own
//! [`SeatView`](engine::view::SeatView) and the legal actions, whether it runs in the same program
//! ([`play_match`]) or over a byte stream ([`serve_match`] on the table side, [`run_seat`] on the
//! seat side).
//!
//! - [`table`]: the in-process table and the seed rules.
//! - [`record`]: match records, their replay and verification, and JSON Lines files.
//! - [`protocol`]: the versioned line protocol and its in-memory and TCP transports.
//! - [`leak`]: the card-set checks used by the leakage tests (what a seat may and may not see).

pub mod leak;
pub mod protocol;
pub mod record;
pub mod table;

pub use protocol::{
    MemLink, PROTOCOL_VERSION, SeatError, SeatLink, SeatMsg, TableMsg, TcpLink, mem_link_pair,
    run_seat, serve_match,
};
pub use record::{
    FORMAT_VERSION, GameSummary, MatchRecord, RecordError, RecordReader, RecordWriter, TrickPoint,
};
pub use table::{
    BlockSpec, SEAT_SEED_LABEL, TableConfig, TableError, build_match, drive_match, play_match,
    seat_seed,
};
