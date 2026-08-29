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
    /// Per-neuron spike-count / firing-rate features only (`len == n_neurons`).
    #[default]
    RateSum,
    /// Time-binned spike histogram features only (`len == n_neurons * 4`).
    TemporalHistogram,
    /// Clamped membrane-potential features only (`len == n_neurons`).
    MembraneSnapshot,
    /// Pure-path alias of [`RateSum`]; full ternary/GIF impl is out of crate.
    SpikingTernary,
}

/// MoE-aware precision tier for **dry-run planning** — not a runtime dtype.
///
/// Tier names align with grok-ozempic `TensorPrecision` (`src/types.rs`) and the
/// `xai-dissect` manifest vocabulary. hybrid-fusion *plans* with these tiers; it
/// never converts, packs, or writes weights. GOZ1 packing, real quantize paths,
/// and CUDA kernels stay in `grok-ozempic` / `myelin-accelerator`. Consumed
/// through [`HybridStagePlanner`](crate::HybridStagePlanner).
///
/// # Wire names
///
/// The crate's first **wire-vocabulary** type, so unlike in-process knobs such as
/// [`ProjectionMode`] it carries `#[serde(rename_all = "snake_case")]` and emits
/// exactly `"preserve"`, `"fp16"`, `"ternary_snn"`. The research enum omits the
/// attribute and spells the variant `TernarySnN`, emitting `"TernarySnN"`; that
/// divergence is fixed here deliberately. The `Snn` spelling is also the only one
/// `rename_all` maps to `"ternary_snn"` (`TernarySnN` yields `"ternary_sn_n"`).
///
/// # Default
///
/// [`Preserve`](Self::Preserve) — this crate plans but never quantizes, so the
/// safe default is the tier that changes nothing. The research pipeline defaults
/// to `ternary_snn` via `manifest.defaults.precision`; that policy travels with
/// the manifest, which is out of this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionTier {
    /// Routing-critical no-touch tier; serializes as `"preserve"`.
    #[default]
    Preserve,
    /// Keep source FP16 — the tier MoE routing gates plan at; serializes as `"fp16"`.
    Fp16,
    /// Two-bit ternary `{-1, 0, +1}` for the spiking path; serializes as `"ternary_snn"`.
    TernarySnn,
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
    ///
    /// Dense when `Some`: one entry per expert (`len == num_experts`, sum ≈ 1).
    ///
    /// Sourced from [`crate::ExpertRouteOutput`] but **renormalized, not carried
    /// through verbatim**: [`ReverseHybridPath::forward_activity`](crate::ReverseHybridPath::forward_activity)
    /// rejects a router weight sum outside [`crate::WEIGHT_SUM_TOLERANCE`], then
    /// rescales the accepted weights by `1 / sum` in `f64`. The rescale is exactly
    /// a no-op when the router already sums to `1.0`; otherwise each weight moves
    /// by a relative amount on the order of that tolerance.
    /// [`routing_entropy`](Self::routing_entropy) is **not** recomputed, so it
    /// still describes the router's pre-rescale distribution.
    pub expert_weights: Option<Vec<f32>>,
    /// Selected expert indices when reverse-path routing ran; `None` otherwise.
    ///
    /// Sparse when `Some`: only the top-k indices (`len == top_k`, each
    /// `< num_experts`) — not aligned element-wise with `expert_weights`.
    pub selected_experts: Option<Vec<usize>>,
    /// Optional routing entropy telemetry from an `ExpertRouter`.
    ///
    /// Passed through exactly as the router reported it; it is **not** recomputed
    /// after [`expert_weights`](Self::expert_weights) renormalization.
    pub routing_entropy: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tier names are a wire vocabulary shared with `xai-dissect` manifests and
    /// grok-ozempic CLI strings; `rename_all` is what keeps them aligned. Without
    /// it serde emits `"TernarySnn"`, and the research spelling `TernarySnN` would
    /// emit `"ternary_sn_n"`.
    #[test]
    fn precision_tier_serializes_to_manifest_names() {
        assert_eq!(
            serde_json::to_string(&PrecisionTier::Preserve).unwrap(),
            "\"preserve\""
        );
        assert_eq!(
            serde_json::to_string(&PrecisionTier::Fp16).unwrap(),
            "\"fp16\""
        );
        assert_eq!(
            serde_json::to_string(&PrecisionTier::TernarySnn).unwrap(),
            "\"ternary_snn\""
        );
    }

    #[test]
    fn precision_tier_round_trips_through_json() {
        for tier in [
            PrecisionTier::Preserve,
            PrecisionTier::Fp16,
            PrecisionTier::TernarySnn,
        ] {
            let json = serde_json::to_string(&tier).unwrap();
            assert_eq!(serde_json::from_str::<PrecisionTier>(&json).unwrap(), tier);
        }
    }

    /// This crate plans, it never quantizes, so the default must be the tier that
    /// changes nothing — inverting the research pipeline's `ternary_snn`.
    #[test]
    fn precision_tier_default_is_preserve() {
        assert_eq!(PrecisionTier::default(), PrecisionTier::Preserve);
    }
}
