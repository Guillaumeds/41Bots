//! The student network (docs/STUDENT.md): a small network that sees only what one seat can see
//! and gives every legal card a money-chance value, trained to copy the teacher (PIMC over many
//! deals, each searched with the hybrid H_k).
//!
//! - [`position`]: a decision position in raw form (start state, deck, actions, seat), replayed
//!   to the seat's exact view; the candidates' codes.
//! - [`record`]: the teacher's stored records (format `41bots-student-teacher-v1`): every
//!   candidate's money chance in every deal, with the stamps; append-only, content-addressed.
//!
//! Everything that scores is fixed point ([`equity::MoneyChance`]); floats appear only inside
//! training (Python).

pub mod position;
pub mod record;

/// The seed of the fixed test set (docs/STUDENT.md §7): the generator refuses it for training
/// shards, and the trainer refuses shards made with it.
pub const STUDENT_TEST_SEED: u64 = 0x7E57;
