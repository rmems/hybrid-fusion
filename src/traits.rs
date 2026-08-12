// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::Result;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

pub trait Transformer {
    fn hidden_states(&self, token_ids: &[u32]) -> Tensor;
    fn dim(&self) -> usize;
    fn max_seq_len(&self) -> usize;
    fn param_count(&self) -> usize;
}

pub trait SpikingNetwork {
    fn step(&mut self, stimuli: &[f32], modulators: &NeuroModulators) -> Result<Vec<usize>>;
    fn num_channels(&self) -> usize;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuroModulators {
    pub dopamine: f32,
    pub cortisol: f32,
    pub acetylcholine: f32,
    pub tempo: f32,
    pub aux_dopamine: f32,
}

impl Default for NeuroModulators {
    fn default() -> Self {
        Self {
            dopamine: 0.5,
            cortisol: 0.0,
            acetylcholine: 0.5,
            tempo: 1.0,
            aux_dopamine: 0.0,
        }
    }
}

pub trait GgufLoader {
    fn load(&self, path: &str) -> Result<GgufLayout>;
}

pub struct GgufLayout {
    pub architecture: String,
    pub tensor_count: usize,
}

// ── Reverse path: SNN activity → (project) → MoE ExpertRouter ───────────────
// Shapes inspired by corinth-canal FunnelActivity / Router::forward / RouterOutput.
// No GIF dynamics, checkpoint load, or family adapters live here.

/// Pure spike / membrane surface for reverse-path projection (no neuron dynamics).
///
/// Field names mirror corinth-canal `FunnelActivity` / `Projector::project` inputs
/// (`spike_train`, `potentials`, `iz_potentials`) without ternary events or GIF state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpikeActivity {
    /// Per-timestep fired neuron indices.
    pub spike_train: Vec<Vec<usize>>,
    /// Membrane potentials (typically length = neuron count).
    pub potentials: Vec<f32>,
    /// Optional adaptive-bank potentials; empty when unused.
    pub iz_potentials: Vec<f32>,
}

impl SpikeActivity {
    /// Build a one-step activity bag from fired indices (zeros membranes).
    pub fn from_fired(fired: &[usize], n_neurons: usize) -> Self {
        Self {
            spike_train: vec![fired.to_vec()],
            potentials: vec![0.0; n_neurons],
            iz_potentials: Vec::new(),
        }
    }
}

/// Result of MoE expert routing (corinth-canal `RouterOutput` subset + optional entropy).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertRouteOutput {
    /// Full gate distribution over experts (length == `num_experts` when normalized).
    pub expert_weights: Vec<f32>,
    /// Selected expert indices (typically length == `top_k`).
    pub selected_experts: Vec<usize>,
    /// Optional routing entropy for telemetry; `None` if not computed.
    pub routing_entropy: Option<f32>,
}

/// Mixture-of-Experts routing contract: embedding → expert weights + selection.
///
/// Concrete checkpoint-backed routers live outside this crate. Reference
/// implementations (e.g. uniform stub) may ship under the `backends` feature.
///
/// Embedding length is **not** fixed to a research constant (corinth uses 2048);
/// backends validate dimensions they care about.
pub trait ExpertRouter {
    fn num_experts(&self) -> usize;
    fn top_k(&self) -> usize;
    fn route(&mut self, embedding: &[f32]) -> Result<ExpertRouteOutput>;
}
