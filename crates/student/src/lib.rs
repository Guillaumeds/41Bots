//! The student network (docs/STUDENT.md): a small network that sees only what one seat can see
//! and gives every legal card a money-chance value, trained to copy the teacher (PIMC over many
//! deals, each searched with the hybrid H_k).
//!
//! - [`position`]: a decision position in raw form (start state, deck, actions, seat), replayed
//!   to the seat's exact view; the candidates' codes.
//! - [`record`]: the teacher's stored records (format `41bots-student-teacher-v1`): every
//!   candidate's money chance in every deal, with the stamps; append-only, content-addressed.
//! - [`encode`]: the encoder `student-v1` (the single source of truth): a seat's view and a
//!   candidate as feature indices; its versioned hash.
//! - [`features`]: feature shards (format `41bots-student-features-v1`), what Python trains on.
//! - [`data`]: teacher shards read together: stamps checked, decisions merged by deal index,
//!   encoded as feature rows; the deal re-weighting hook.
//! - [`net`]: the integer student network (the oracle's integer scheme), one scorer for every
//!   candidate, and the bot's choice.
//! - [`check`]: the student against the teacher's stored values: decision loss and agreement.
//!
//! Everything that scores is fixed point ([`equity::MoneyChance`]); floats appear only inside
//! training (Python).

pub mod check;
pub mod data;
pub mod encode;
pub mod features;
pub mod net;
pub mod position;
pub mod record;

/// The seed of the fixed test set (docs/STUDENT.md §7): the generator refuses it for training
/// shards, and the trainer refuses shards made with it.
pub const STUDENT_TEST_SEED: u64 = 0x7E57;
