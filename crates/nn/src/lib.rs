//! The oracle value networks of the hybrid yardstick (docs/M3-HYBRID-SPEC.md §6; PLAN §6.5 head
//! 4 and §13 `crates/nn`).
//!
//! - [`encode`]: the position encoder, the single source of truth. It turns the solver's
//!   canonical form of a position at a trick boundary ([`solver::CanonPos`]) plus the match
//!   context into a list of active feature indices and a few dense values. Python never
//!   re-implements it: it reads encoded features from the label shards.
//! - [`mlp`]: integer-only MLP inference (fixed-point weights and activations, integer
//!   accumulators, ReLU clipped at 64), four output logits: the plain reference.
//! - [`fast`]: the same forward pass, bit for bit, with less work (sparse and incremental first
//!   layer, vectorised kernels); [`kernel`]: its inner loops, compiled for x86-64-v2/v3/v4 with
//!   runtime dispatch.
//! - [`money`]: the fixed integer rule from four logits to a [`equity::MoneyChance`] summing to
//!   exactly 2^32 (an integer exponential, then largest-remainder normalisation).
//! - [`safetensors`]: a tiny reader and writer of safetensors files (the weights format).
//! - [`shard`]: label shards, `.npy` files of fixed little-endian records (PLAN §12.7).
//! - [`outcome`]: the outcome head (spec §6a): how the hand can end from a position, the exact
//!   money vector of each ending, and the value of a distribution over them.
//! - [`oracle`]: a loaded network as a solver frontier evaluator ([`solver::Frontier`]).
//! - [`synthetic`]: a synthetic leaf, network and features for tests and fixtures.
//!
//! Everything here is integer arithmetic, so a network's money chances are byte-identical on
//! every machine, thread count and WASM; floats exist only in training (Python).

pub mod encode;
pub mod fast;
pub mod kernel;
pub mod mlp;
pub mod money;
pub mod oracle;
pub mod outcome;
pub mod safetensors;
pub mod shard;
pub mod synthetic;

pub use encode::{Features, encode};
pub use mlp::IntMlp;
pub use money::money_from_logits;
pub use oracle::Oracle;
