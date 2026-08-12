// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

/// Strategy to convert [`SpikeActivity`](crate::SpikeActivity) into a dense embedding
/// for reverse-path **MoE** [`ExpertRouter`](crate::ExpertRouter) (not SAAQ).
///
/// Pure feature construction (no learned W/b). [`SpikingTernary`](Self::SpikingTernary)
/// is a trait-level hook: pure path matches [`RateSum`](Self::RateSum); GIF membrane
/// dynamics stay in `neuromod` / research.
///
/// Source shape: corinth-canal `ProjectionMode` (`src/types.rs` / `src/projector.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ProjectionMode {
    /// Per-neuron firing rates + temporal bins + membrane (+ optional iz bank).
    #[default]
    RateSum,
    /// Emphasizes time-binned spike histograms over raw rates.
    TemporalHistogram,
    /// Emphasizes membrane potentials as the primary signal.
    MembraneSnapshot,
    /// Pure-path alias of RateSum features; full ternary/GIF impl is out of crate.
    SpikingTernary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub ff_dim: usize,
    pub max_seq_len: usize,
}

impl TransformerConfig {
    pub fn tiny() -> Self {
        Self {
            vocab_size: 256,
            dim: 128,
            num_heads: 4,
            num_layers: 2,
            ff_dim: 256,
            max_seq_len: 64,
        }
    }

    pub fn olmo_1b() -> Self {
        Self {
            vocab_size: 50304,
            dim: 2048,
            num_heads: 16,
            num_layers: 22,
            ff_dim: 5504,
            max_seq_len: 2048,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    pub transformer: TransformerConfig,
    pub snn_lif_neurons: usize,
    pub snn_izh_neurons: usize,
    pub snn_input_channels: usize,
}

impl HybridConfig {
    pub fn tiny() -> Self {
        Self {
            transformer: TransformerConfig::tiny(),
            snn_lif_neurons: 32,
            snn_izh_neurons: 8,
            snn_input_channels: 64,
        }
    }

    pub fn olmo_1b() -> Self {
        Self {
            transformer: TransformerConfig::olmo_1b(),
            snn_lif_neurons: 128,
            snn_izh_neurons: 32,
            snn_input_channels: 256,
        }
    }
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self::tiny()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridOutput {
    pub embedding: Vec<f32>,
    pub stimuli: Vec<f32>,
    pub fired_neurons: Vec<usize>,
    pub global_step: u64,
    /// MoE gate weights when reverse-path routing ran; `None` on ANN→SNN only.
    pub expert_weights: Option<Vec<f32>>,
    /// Selected expert indices when reverse-path routing ran; `None` otherwise.
    pub selected_experts: Option<Vec<usize>>,
    /// Optional routing entropy telemetry from an `ExpertRouter`.
    pub routing_entropy: Option<f32>,
}
