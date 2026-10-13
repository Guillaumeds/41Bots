//! The networked table (docs/M1-SPEC.md §3.4): versioned JSON messages, one per line, over a byte
//! stream.
//!
//! Table to seat:
//! - `{"v":1,"type":"hello","seat":3,"rules":{...},"seed":123}`: the seat number, the rules and
//!   the seed for the bot's `start_match` (the seat seed, never the match seed, which would give
//!   away the deal).
//! - `{"v":1,"type":"act","view":{SeatView},"legal":[{"Play":17},...]}`: the seat must act.
//! - `{"v":1,"type":"end","record":{MatchRecord}}`: the match is over.
//! - `{"v":1,"type":"abort","reason":"..."}`: the match was stopped (the reason names no card).
//!
//! Seat to table:
//! - `{"v":1,"type":"ready","name":"rules-v1"}`: the answer to `hello`, naming the agent.
//! - `{"v":1,"type":"action","action":{"Play":17}}`: the answer to `act`.

use bot::Bot;
use engine::SEATS;
use engine::game::{Action, Rules};
use engine::view::SeatView;
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::sync::mpsc::{Receiver, Sender, channel};

use crate::record::MatchRecord;
use crate::table::{TableConfig, TableError, drive_match, seat_seed};

/// The protocol version every message carries as `"v"`.
pub const PROTOCOL_VERSION: u32 = 1;

/// The longest line a [`TcpLink`] accepts (16 MiB); a longer one closes the link.
pub const MAX_LINE_BYTES: u64 = 16 << 20;

/// A message from the table to a seat.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum TableMsg {
    /// The match is starting: this is your seat, the rules and your bot's seed.
    Hello {
        /// The seat, 0..=7.
        seat: u8,
        /// The rule switches.
        rules: Rules,
        /// The seed for `Bot::start_match` (the seat seed of docs/M1-SPEC.md §1).
        seed: u64,
    },
    /// Your seat must act: choose one of `legal`.
    Act {
        /// Your seat's view.
        view: Box<SeatView>,
        /// The legal actions, sorted by card index.
        legal: Vec<Action>,
    },
    /// The match is over.
    End {
        /// The match record.
        record: Box<MatchRecord>,
    },
    /// The match was stopped before its end.
    Abort {
        /// Why, in plain English, naming no card.
        reason: String,
    },
}

/// A message from a seat to the table.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SeatMsg {
    /// The answer to `hello`: the agent's name, for the record.
    Ready {
        /// The agent's name.
        name: String,
    },
    /// The answer to `act`.
    Action {
        /// The chosen action.
        action: Action,
    },
}

#[derive(Serialize)]
struct EnvelopeOut<'a, T> {
    v: u32,
    #[serde(flatten)]
    msg: &'a T,
}

#[derive(Deserialize)]
struct EnvelopeIn<T> {
    v: u32,
    #[serde(flatten)]
    msg: T,
}

#[derive(Deserialize)]
struct VersionOnly {
    v: u32,
}

fn encode<T: Serialize>(msg: &T) -> String {
    serde_json::to_string(&EnvelopeOut {
        v: PROTOCOL_VERSION,
        msg,
    })
    .expect("protocol messages always serialise")
}

fn decode<T: for<'de> Deserialize<'de>>(line: &str) -> Result<T, String> {
    match serde_json::from_str::<EnvelopeIn<T>>(line) {
        Ok(env) if env.v == PROTOCOL_VERSION => Ok(env.msg),
        Ok(env) => Err(format!(
            "protocol version {} is not supported (this table speaks version {PROTOCOL_VERSION})",
            env.v
        )),
        Err(e) => match serde_json::from_str::<VersionOnly>(line) {
            Ok(v) if v.v != PROTOCOL_VERSION => Err(format!(
                "protocol version {} is not supported (this table speaks version {PROTOCOL_VERSION})",
                v.v
            )),
            _ => Err(format!("not a valid message: {e}")),
        },
    }
}

/// One table-to-seat message as a line (no newline).
pub fn encode_table(msg: &TableMsg) -> String {
    encode(msg)
}

/// Parse a table-to-seat line.
pub fn decode_table(line: &str) -> Result<TableMsg, String> {
    decode(line)
}

/// One seat-to-table message as a line (no newline).
pub fn encode_seat(msg: &SeatMsg) -> String {
    encode(msg)
}

/// Parse a seat-to-table line.
pub fn decode_seat(line: &str) -> Result<SeatMsg, String> {
    decode(line)
}

/// A two-way line transport between the table and one seat. Lines carry no newline.
pub trait SeatLink {
    /// Send one line. A failure is not reported here; the next [`SeatLink::recv`] returns `None`.
    fn send(&mut self, line: &str);
    /// The next line from the other side, waiting for it; `None` once the link is closed or has
    /// failed.
    fn recv(&mut self) -> Option<String>;
}

impl<L: SeatLink + ?Sized> SeatLink for Box<L> {
    fn send(&mut self, line: &str) {
        (**self).send(line);
    }
    fn recv(&mut self) -> Option<String> {
        (**self).recv()
    }
}

/// One end of an in-memory link (`std::sync::mpsc` channels), for tests and in-process use.
pub struct MemLink {
    tx: Sender<String>,
    rx: Receiver<String>,
}

/// A connected pair of in-memory link ends: give one to the table and the other to the seat.
pub fn mem_link_pair() -> (MemLink, MemLink) {
    let (tx_a, rx_b) = channel();
    let (tx_b, rx_a) = channel();
    (
        MemLink { tx: tx_a, rx: rx_a },
        MemLink { tx: tx_b, rx: rx_b },
    )
}

impl SeatLink for MemLink {
    fn send(&mut self, line: &str) {
        // A closed other end shows up as `None` from the next `recv`.
        let _ = self.tx.send(line.to_string());
    }
    fn recv(&mut self) -> Option<String> {
        self.rx.recv().ok()
    }
}

/// A TCP link (`std::net`, blocking), one line per message. Nagle's algorithm is turned off, since
/// every message waits for an answer.
pub struct TcpLink {
    reader: BufReader<TcpStream>,
    writer: TcpStream,
    broken: bool,
}

impl TcpLink {
    /// Wrap a connected stream. Set any read timeout on the stream first.
    pub fn new(stream: TcpStream) -> io::Result<TcpLink> {
        stream.set_nodelay(true)?;
        let writer = stream.try_clone()?;
        Ok(TcpLink {
            reader: BufReader::new(stream),
            writer,
            broken: false,
        })
    }

    /// Connect to a table.
    pub fn connect(addr: impl ToSocketAddrs) -> io::Result<TcpLink> {
        TcpLink::new(TcpStream::connect(addr)?)
    }

    /// Accept the next seat's connection.
    pub fn accept(listener: &TcpListener) -> io::Result<TcpLink> {
        let (stream, _) = listener.accept()?;
        TcpLink::new(stream)
    }
}

impl SeatLink for TcpLink {
    fn send(&mut self, line: &str) {
        if self.broken {
            return;
        }
        let mut buf = String::with_capacity(line.len() + 1);
        buf.push_str(line);
        buf.push('\n');
        if self
            .writer
            .write_all(buf.as_bytes())
            .and_then(|()| self.writer.flush())
            .is_err()
        {
            self.broken = true;
        }
    }

    fn recv(&mut self) -> Option<String> {
        if self.broken {
            return None;
        }
        let mut line = String::new();
        match (&mut self.reader).take(MAX_LINE_BYTES).read_line(&mut line) {
            Ok(_) if line.ends_with('\n') => {
                line.pop();
                if line.ends_with('\r') {
                    line.pop();
                }
                Some(line)
            }
            // End of stream, a line cut off by it, an over-long line, or an error.
            _ => {
                self.broken = true;
                None
            }
        }
    }
}

/// Why a seat's client stopped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SeatError {
    /// The link closed before the match ended.
    Disconnected,
    /// The table sent something the protocol does not allow.
    Protocol(String),
    /// The table stopped the match.
    Aborted(String),
}

impl core::fmt::Display for SeatError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SeatError::Disconnected => write!(f, "the table disconnected"),
            SeatError::Protocol(d) => write!(f, "the table broke the protocol: {d}"),
            SeatError::Aborted(r) => write!(f, "the table stopped the match: {r}"),
        }
    }
}

impl std::error::Error for SeatError {}

fn recv_seat(link: &mut dyn SeatLink, seat: u8) -> Result<SeatMsg, TableError> {
    let line = link.recv().ok_or(TableError::Disconnected { seat })?;
    decode_seat(&line).map_err(|detail| TableError::Protocol { seat, detail })
}

/// Run one match over eight links, link `s` being seat `s` (the table side of
/// docs/M1-SPEC.md §3.4). Sends `hello` to every seat and waits for each `ready`, then plays the
/// match exactly as [`crate::play_match`] does, sending each decision's view and legal actions
/// to the seat to act only. At the end every seat receives `end` with the record; if the match is
/// stopped, every seat receives `abort` and the error is returned.
pub fn serve_match(
    cfg: &TableConfig,
    mut links: [Box<dyn SeatLink>; SEATS],
) -> Result<MatchRecord, TableError> {
    let result = serve(cfg, &mut links);
    let last = match &result {
        Ok(record) => TableMsg::End {
            record: Box::new(record.clone()),
        },
        Err(e) => TableMsg::Abort {
            reason: e.public_reason(),
        },
    };
    let line = encode_table(&last);
    for link in &mut links {
        link.send(&line);
    }
    result
}

fn serve(
    cfg: &TableConfig,
    links: &mut [Box<dyn SeatLink>; SEATS],
) -> Result<MatchRecord, TableError> {
    for (seat, link) in (0u8..).zip(links.iter_mut()) {
        link.send(&encode_table(&TableMsg::Hello {
            seat,
            rules: cfg.rules,
            seed: seat_seed(cfg.seed, seat),
        }));
    }
    let mut names: [String; SEATS] = core::array::from_fn(|_| String::new());
    for (seat, link) in (0u8..).zip(links.iter_mut()) {
        match recv_seat(link.as_mut(), seat)? {
            SeatMsg::Ready { name } => names[usize::from(seat)] = name,
            SeatMsg::Action { .. } => {
                return Err(TableError::Protocol {
                    seat,
                    detail: "sent an action before answering hello with ready".to_string(),
                });
            }
        }
    }
    let mut m = cfg.new_match();
    drive_match(&mut m, &names, |seat, view, legal| {
        let link = links[usize::from(seat)].as_mut();
        link.send(&encode_table(&TableMsg::Act {
            view: Box::new(view.clone()),
            legal: legal.to_vec(),
        }));
        match recv_seat(link, seat)? {
            SeatMsg::Action { action } => Ok(action),
            SeatMsg::Ready { .. } => Err(TableError::Protocol {
                seat,
                detail: "sent ready when an action was due".to_string(),
            }),
        }
    })?;
    Ok(MatchRecord::from_match(&m, cfg.seed, cfg.block, names)
        .expect("drive_match returns Ok only once the match is over"))
}

/// Play one match as a seat over a link (the seat side of docs/M1-SPEC.md §3.4): answer `hello`
/// by starting the bot and sending `ready`, answer each `act` with the bot's action, and return
/// the record from `end`. The link can be used again for another match.
pub fn run_seat(link: &mut dyn SeatLink, bot: &mut dyn Bot) -> Result<MatchRecord, SeatError> {
    let mut my_seat: Option<u8> = None;
    loop {
        let line = link.recv().ok_or(SeatError::Disconnected)?;
        match decode_table(&line).map_err(SeatError::Protocol)? {
            TableMsg::Hello { seat, rules, seed } => {
                if usize::from(seat) >= SEATS {
                    return Err(SeatError::Protocol(format!("seat {seat} does not exist")));
                }
                bot.start_match(seat, rules, seed);
                my_seat = Some(seat);
                link.send(&encode_seat(&SeatMsg::Ready {
                    name: bot.name().to_string(),
                }));
            }
            TableMsg::Act { view, legal } => {
                let Some(seat) = my_seat else {
                    return Err(SeatError::Protocol("act before hello".to_string()));
                };
                if view.seat != seat || view.public.to_act != Some(seat) || legal.is_empty() {
                    return Err(SeatError::Protocol(
                        "act for another seat, or with no legal action".to_string(),
                    ));
                }
                let action = bot.act(&view, &legal);
                link.send(&encode_seat(&SeatMsg::Action { action }));
            }
            TableMsg::End { record } => return Ok(*record),
            TableMsg::Abort { reason } => return Err(SeatError::Aborted(reason)),
        }
    }
}
