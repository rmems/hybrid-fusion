// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reference backend implementations for the Transformer, SpikingNetwork, and
//! ExpertRouter traits.
//!
//! These are minimal, deterministic implementations intended for testing,
//! prototyping, and as a working example of the trait contracts. They are
//! **not** optimised for production inference.

pub mod simple_snn;
pub mod simple_transformer;
pub mod stub_router;
