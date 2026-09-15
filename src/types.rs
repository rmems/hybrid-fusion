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

/// Storage dtype of a checkpoint tensor (Safetensors header vocabulary).
///
/// Metadata only: this crate never parses payloads or memmaps files. Concrete
/// header decode belongs in `engram-parser`. Wire names match the Safetensors
/// spec (`"F32"`, `"BF16"`, `"BOOL"`, …).
///
/// Source shape: Hugging Face Safetensors `dtype` strings; consumed by
/// [`TensorManifestEntry`] and [`SafetensorsLayout`](crate::SafetensorsLayout).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Dtype {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

/// MoE-oriented role of a named checkpoint tensor.
///
/// Heuristic classification for ExpertRouter / dry-run planner / routing-math
/// consumers. Name-only — no file I/O, no family adapters, no payload inspect.
/// [`Other`](Self::Other) is the safe default when a name does not match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TensorRole {
    /// MoE router / gate tensor (`…gate.weight`, `…router…`).
    Router,
    /// Expert parameter tensor (`…experts.{i}…`, `shared_expert`, …).
    ExpertWeight,
    /// Attention projection / scores (`self_attn`, `q_proj`, …).
    Attention,
    /// Token / position embeddings and output head (`embed_tokens`, `lm_head`).
    Embedding,
    /// Normalization (`layernorm`, `rms_norm`, `*.norm.*`).
    Norm,
    /// Unclassified (FFN `up_proj` / `gate_proj`, biases, unknown families).
    #[default]
    Other,
}

impl TensorRole {
    /// Infer a role from a checkpoint tensor **name**. Name heuristics only;
    /// `shape` is not consulted (reserved for parser-side validation).
    ///
    /// Matching is case-insensitive and uses `.`-separated path components, so
    /// FFN `gate_proj` is **not** a router (unlike Mixtral/Qwen `mlp.gate`).
    /// Expert components win over gate/router so `experts.0.gate_proj` is
    /// [`ExpertWeight`](Self::ExpertWeight).
    pub fn from_name(name: &str) -> Self {
        let n = name.to_ascii_lowercase();
        let parts: Vec<&str> = n.split('.').collect();

        if parts.iter().any(|p| p.contains("expert")) {
            return Self::ExpertWeight;
        }
        if parts
            .iter()
            .any(|p| *p == "gate" || *p == "router" || p.starts_with("router"))
        {
            return Self::Router;
        }
        if is_attention_component(&parts) {
            return Self::Attention;
        }
        if is_embedding_component(&parts) {
            return Self::Embedding;
        }
        if is_norm_component(&parts) {
            return Self::Norm;
        }
        Self::Other
    }
}

fn is_attention_component(parts: &[&str]) -> bool {
    const ATTN: &[&str] = &[
        "self_attn",
        "attn",
        "attention",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qkv",
        "wq",
        "wk",
        "wv",
        "wo",
    ];
    parts.iter().any(|p| ATTN.contains(p) || p.contains("attn"))
}

fn is_embedding_component(parts: &[&str]) -> bool {
    const EMBED: &[&str] = &[
        "embed_tokens",
        "embed",
        "embeddings",
        "wte",
        "wpe",
        "lm_head",
        "tok_embeddings",
        "token_embd",
    ];
    parts
        .iter()
        .any(|p| EMBED.contains(p) || p.contains("embed"))
}

fn is_norm_component(parts: &[&str]) -> bool {
    parts.iter().any(|p| p.contains("norm"))
}

/// One named tensor in a Safetensors (or HF-sharded) inventory.
///
/// Layout metadata only — no payload bytes. `shape` uses the same `Vec<usize>`
/// convention as [`crate::Tensor::shape`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorManifestEntry {
    /// Checkpoint tensor name (e.g. `model.layers.0.mlp.gate.weight`).
    pub name: String,
    /// Header storage dtype (not a dry-run precision-planning policy).
    pub dtype: Dtype,
    /// Dimension extents; every axis should be `> 0` (same invariant as `Tensor`).
    pub shape: Vec<usize>,
    /// Source shard filename from a Hugging Face `model.safetensors.index.json`
    /// weight map, if the checkpoint is sharded.
    pub shard: Option<String>,
    /// MoE-oriented role, typically from [`TensorRole::from_name`].
    pub role: TensorRole,
    /// Optional free-form labels (architecture tags, layer ids, …). Empty if unused.
    pub labels: Vec<String>,
}

impl TensorManifestEntry {
    /// Build an entry and infer [`role`](Self::role) from `name`.
    pub fn new(
        name: String,
        dtype: Dtype,
        shape: Vec<usize>,
        shard: Option<String>,
        labels: Vec<String>,
    ) -> Self {
        let role = TensorRole::from_name(&name);
        Self {
            name,
            dtype,
            shape,
            shard,
            role,
            labels,
        }
    }
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

    #[test]
    fn tensor_role_default_is_other() {
        assert_eq!(TensorRole::default(), TensorRole::Other);
    }

    #[test]
    fn tensor_role_from_name_router_gate() {
        assert_eq!(
            TensorRole::from_name("model.layers.0.block_sparse_moe.gate.weight"),
            TensorRole::Router
        );
        assert_eq!(
            TensorRole::from_name("model.layers.0.mlp.gate.weight"),
            TensorRole::Router
        );
        assert_eq!(
            TensorRole::from_name("model.layers.0.mlp.router.weight"),
            TensorRole::Router
        );
    }

    #[test]
    fn tensor_role_from_name_expert_weight() {
        assert_eq!(
            TensorRole::from_name("model.layers.0.block_sparse_moe.experts.0.w1.weight"),
            TensorRole::ExpertWeight
        );
        // Expert components win over gate so SwiGLU-inside-expert is not Router.
        assert_eq!(
            TensorRole::from_name("model.layers.0.mlp.experts.3.gate_proj.weight"),
            TensorRole::ExpertWeight
        );
    }

    #[test]
    fn tensor_role_from_name_attention() {
        assert_eq!(
            TensorRole::from_name("model.layers.0.self_attn.q_proj.weight"),
            TensorRole::Attention
        );
    }

    #[test]
    fn tensor_role_from_name_unknown_and_ffn_gate_proj_are_other() {
        assert_eq!(
            TensorRole::from_name("model.layers.0.mlp.up_proj.weight"),
            TensorRole::Other
        );
        // Dense FFN SwiGLU gate — not an MoE router.
        assert_eq!(
            TensorRole::from_name("model.layers.0.mlp.gate_proj.weight"),
            TensorRole::Other
        );
    }

    #[test]
    fn tensor_role_from_name_embedding_and_norm() {
        assert_eq!(
            TensorRole::from_name("model.embed_tokens.weight"),
            TensorRole::Embedding
        );
        assert_eq!(
            TensorRole::from_name("model.layers.0.input_layernorm.weight"),
            TensorRole::Norm
        );
    }

    #[test]
    fn dtype_serializes_to_safetensors_header_names() {
        assert_eq!(serde_json::to_string(&Dtype::F32).unwrap(), "\"F32\"");
        assert_eq!(serde_json::to_string(&Dtype::BF16).unwrap(), "\"BF16\"");
        assert_eq!(serde_json::to_string(&Dtype::Bool).unwrap(), "\"BOOL\"");
    }

    #[test]
    fn tensor_manifest_entry_new_infers_role() {
        let e = TensorManifestEntry::new(
            "model.layers.0.mlp.gate.weight".into(),
            Dtype::F16,
            vec![8, 16],
            Some("model-00001-of-00002.safetensors".into()),
            vec!["moe".into()],
        );
        assert_eq!(e.role, TensorRole::Router);
        assert_eq!(e.dtype, Dtype::F16);
        assert_eq!(e.shape, vec![8, 16]);
    }
}
