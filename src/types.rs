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
    ///
    /// Each rate is the per-neuron mean count over timesteps, in `[0, 1]`
    /// (a timestep is a set; see
    /// [`SpikeActivity`](crate::SpikeActivity) "Timestep semantics").
    #[default]
    RateSum,
    /// Time-binned spike histogram features only (`len == n_neurons * 4`).
    TemporalHistogram,
    /// Clamped membrane-potential features only (`len == n_neurons`).
    MembraneSnapshot,
    /// **Alias** of [`RateSum`](Self::RateSum) (issue #25): on the pure path
    /// this mode emits **identical** firing-rate features and encodes no
    /// ternary (spike/burst/silent) events. It exists as a trait-level hook
    /// for the neuromod GIF bank; GIF/ternary dynamics are out of crate (a
    /// non-extract, `docs/extraction-map.md`) and no mode in this crate
    /// reads `SpikeActivity::iz_potentials`. Until that lands, selecting this
    /// mode is equivalent to selecting `RateSum` — do not expect ternary
    /// encoding.
    #[doc(alias = "RateSum")]
    SpikingTernary,
}

/// Storage dtype of a checkpoint tensor (Safetensors header vocabulary).
///
/// Metadata only: this crate never parses payloads or memmaps files. Concrete
/// header decode belongs in `engram-parser`. Wire names match the Safetensors
/// spec (`"F32"`, `"BF16"`, `"BOOL"`, `"U16"`, `"F8_E4M3"`, …).
///
/// Source shape: Hugging Face Safetensors `dtype` strings from
/// `safetensors::tensor::Dtype`; consumed by [`TensorManifestEntry`] and
/// [`SafetensorsLayout`](crate::SafetensorsLayout).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
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
    U16,
    U32,
    U64,
    Bool,
    F4,
    #[serde(rename = "F6_E2M3")]
    F6E2M3,
    #[serde(rename = "F6_E3M2")]
    F6E3M2,
    #[serde(rename = "F8_E5M2")]
    F8E5M2,
    #[serde(rename = "F8_E4M3")]
    F8E4M3,
    #[serde(rename = "F8_E8M0")]
    F8E8M0,
    #[serde(rename = "F8_E4M3FNUZ")]
    F8E4M3Fnuz,
    #[serde(rename = "F8_E5M2FNUZ")]
    F8E5M2Fnuz,
    C64,
}

/// MoE-oriented role of a named checkpoint tensor.
///
/// Legacy inventory helper. Topology and tensor-role discovery move to
/// engram-parser ([#88](https://github.com/rmems/engram-parser/issues/88)).
/// This enum stays so existing manifests keep a stable role spelling; it is
/// not the planning input ([`crate::HybridTopology`] is).
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
    /// Normalization (`layernorm`, `rms_norm`, `*.norm.*`, GPT-style `ln_*`).
    Norm,
    /// Unclassified (FFN `up_proj` / `gate_proj`, biases, unknown families).
    #[default]
    Other,
}

impl TensorRole {
    /// Infer a role from a checkpoint tensor **name**.
    ///
    /// Legacy name-only classifier. Topology and role discovery move to
    /// engram-parser ([#88](https://github.com/rmems/engram-parser/issues/88)).
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
        if parts.iter().any(|p| *p == "gate" || p.contains("router")) {
            return Self::Router;
        }
        if is_norm_component(&parts) {
            return Self::Norm;
        }
        if is_attention_component(&parts) {
            return Self::Attention;
        }
        if is_embedding_component(&parts) {
            return Self::Embedding;
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
    parts
        .iter()
        .any(|p| ATTN.contains(p) || p.contains("attn") || p.contains("attention"))
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
        "shared",
    ];
    parts
        .iter()
        .any(|p| EMBED.contains(p) || p.contains("embed"))
}

fn is_norm_component(parts: &[&str]) -> bool {
    parts.iter().any(|p| {
        p.contains("norm") || *p == "ln_1" || *p == "ln_2" || *p == "ln_f" || p.starts_with("ln_")
    })
}

/// One named tensor in a Safetensors (or HF-sharded) inventory.
///
/// Legacy inventory helper. Topology and role discovery move to engram-parser
/// ([#88](https://github.com/rmems/engram-parser/issues/88)). This record stays
/// available for existing manifests; it is not a planning input.
///
/// Layout metadata only — no payload bytes. `shape` uses the same `Vec<usize>`
/// convention as [`crate::Tensor::shape`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorManifestEntry {
    /// Checkpoint tensor name (e.g. `model.layers.0.mlp.gate.weight`).
    pub name: String,
    /// Header storage dtype (not a dry-run precision-planning policy).
    pub dtype: Dtype,
    /// Dimension extents from the Safetensors header. Zero-length dimensions
    /// and rank-0 scalar shapes (`[]`) are valid metadata; the runtime
    /// [`crate::Tensor`] axis rule is not applied here.
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

/// Orchestrator configuration for a transformer plus SNN pair.
///
/// [`crate::HybridNetwork::try_new`] requires `transformer.dim`,
/// `transformer.max_seq_len`, and `snn_input_channels` to be non-zero and to
/// match the injected [`crate::Transformer`] / [`crate::SpikingNetwork`].
/// [`crate::HybridNetwork::new`] skips that check (pre-1.0 compatibility).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    pub transformer: TransformerConfig,
    /// LIF population hint for backend construction / config-only plans.
    /// Injected backends are authoritative; the host reads `num_neurons()`.
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

/// One forward-pass record: pooled embedding, projected SNN stimuli, fired
/// output neurons, and the MoE routing fields when a reverse path ran.
///
/// # JSON / serde policy
///
/// `serde_json` maps a non-finite `f32` (`NaN` or ±`Inf`) to JSON `null`, which
/// destroys type and meaning on the wire (`embedding: vec![NaN]` would become
/// `"embedding":[null]`, `routing_entropy: Some(NaN)` would round-trip as
/// `None`). This type therefore **fails closed**: serializing it while
/// `embedding`, `stimuli`, `expert_weights`, or `routing_entropy` holds any
/// non-finite value returns an error naming the field instead of writing
/// `null`. Wrap it with `?` and the error surfaces as
/// [`crate::HybridError::Json`].
///
/// Deserialization is unchanged and needs no symmetric guard, because strict
/// JSON cannot express a non-finite number at all. That makes `null` the wire
/// spelling of `None` for the two optional fields in records written under
/// this policy; pre-policy records that already contain `null` inside
/// `embedding` / `stimuli` / `expert_weights` are rejected by `serde_json`
/// (`invalid type: null, expected f32`). Do not trust a legacy
/// `"routing_entropy": null`: re-derive the value rather than guessing.
#[derive(Debug, Clone, Deserialize)]
pub struct HybridOutput {
    pub embedding: Vec<f32>,
    pub stimuli: Vec<f32>,
    /// Sparse output neuron IDs, each `< num_neurons`; not input-channel IDs.
    pub fired_neurons: Vec<usize>,
    /// Output population, also the neuron axis for `SpikeActivity::from_fired`
    /// and `ReverseHybridPath`. Required in serialized records: older records
    /// must supply their actual topology rather than infer it from input width
    /// or the largest fired ID (silent neurons need representation too).
    pub num_neurons: usize,
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

/// Fails closed on non-finite `f32` values instead of letting `serde_json`
/// write `null` (see [`HybridOutput`] docs).
impl Serialize for HybridOutput {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        crate::serde_finite::check_f32_slice(&self.embedding, "HybridOutput.embedding")?;
        crate::serde_finite::check_f32_slice(&self.stimuli, "HybridOutput.stimuli")?;
        crate::serde_finite::check_opt_f32_slice(
            &self.expert_weights,
            "HybridOutput.expert_weights",
        )?;
        crate::serde_finite::check_opt_f32(self.routing_entropy, "HybridOutput.routing_entropy")?;

        // Field order mirrors the struct declaration, as the derived impl did.
        let mut state = serializer.serialize_struct("HybridOutput", 8)?;
        state.serialize_field("embedding", &self.embedding)?;
        state.serialize_field("stimuli", &self.stimuli)?;
        state.serialize_field("fired_neurons", &self.fired_neurons)?;
        state.serialize_field("num_neurons", &self.num_neurons)?;
        state.serialize_field("global_step", &self.global_step)?;
        state.serialize_field("expert_weights", &self.expert_weights)?;
        state.serialize_field("selected_experts", &self.selected_experts)?;
        state.serialize_field("routing_entropy", &self.routing_entropy)?;
        state.end()
    }
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
        // Hugging Face T5-style SelfAttention.q — no `attn` substring.
        assert_eq!(
            TensorRole::from_name("encoder.block.0.layer.0.SelfAttention.q.weight"),
            TensorRole::Attention
        );
    }

    #[test]
    fn tensor_role_from_name_attn_layer_norm_is_norm() {
        assert_eq!(
            TensorRole::from_name("model.encoder.layers.0.self_attn_layer_norm.weight"),
            TensorRole::Norm
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
        assert_eq!(
            TensorRole::from_name("transformer.h.0.ln_1.weight"),
            TensorRole::Norm
        );
        assert_eq!(
            TensorRole::from_name("transformer.ln_f.weight"),
            TensorRole::Norm
        );
        assert_eq!(
            TensorRole::from_name("shared.weight"),
            TensorRole::Embedding
        );
        assert_eq!(
            TensorRole::from_name("model.layers.0.mlp.moe_router.weight"),
            TensorRole::Router
        );
    }

    #[test]
    fn dtype_serializes_to_safetensors_header_names() {
        let cases = [
            (Dtype::F64, "F64"),
            (Dtype::F32, "F32"),
            (Dtype::F16, "F16"),
            (Dtype::BF16, "BF16"),
            (Dtype::I64, "I64"),
            (Dtype::I32, "I32"),
            (Dtype::I16, "I16"),
            (Dtype::I8, "I8"),
            (Dtype::U8, "U8"),
            (Dtype::U16, "U16"),
            (Dtype::U32, "U32"),
            (Dtype::U64, "U64"),
            (Dtype::Bool, "BOOL"),
            (Dtype::F4, "F4"),
            (Dtype::F6E2M3, "F6_E2M3"),
            (Dtype::F6E3M2, "F6_E3M2"),
            (Dtype::F8E5M2, "F8_E5M2"),
            (Dtype::F8E4M3, "F8_E4M3"),
            (Dtype::F8E8M0, "F8_E8M0"),
            (Dtype::F8E4M3Fnuz, "F8_E4M3FNUZ"),
            (Dtype::F8E5M2Fnuz, "F8_E5M2FNUZ"),
            (Dtype::C64, "C64"),
        ];
        for (dtype, wire) in cases {
            let json = format!("\"{wire}\"");
            assert_eq!(serde_json::to_string(&dtype).unwrap(), json);
            assert_eq!(serde_json::from_str::<Dtype>(&json).unwrap(), dtype);
        }
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

    #[test]
    fn tensor_manifest_entry_accepts_zero_extent_and_rank0_shapes() {
        let empty =
            TensorManifestEntry::new("empty.weight".into(), Dtype::F32, vec![0, 16], None, vec![]);
        assert_eq!(empty.shape, vec![0, 16]);
        let scalar = TensorManifestEntry::new("scalar".into(), Dtype::F32, vec![], None, vec![]);
        assert!(scalar.shape.is_empty());
    }

    // ── HybridOutput JSON policy (RM-1465) ─────────────────────────────────

    /// Finite ANN→SNN-only output (no MoE fields): `None` optionals.
    fn tiny_forward_output() -> HybridOutput {
        HybridOutput {
            embedding: vec![0.125, -0.75, 0.0],
            stimuli: vec![0.5, -0.5],
            fired_neurons: vec![0],
            num_neurons: 2,
            global_step: 1,
            expert_weights: None,
            selected_experts: None,
            routing_entropy: None,
        }
    }

    /// Finite reverse-path-shaped output (dense MoE fields set).
    fn reverse_shaped_output() -> HybridOutput {
        HybridOutput {
            embedding: vec![-0.5, 0.25, 0.0],
            stimuli: Vec::new(),
            fired_neurons: vec![1],
            num_neurons: 4,
            global_step: 7,
            expert_weights: Some(vec![0.5, 0.25, 0.25]),
            selected_experts: Some(vec![0, 2]),
            routing_entropy: Some(0.42),
        }
    }

    /// Lossless round trip: deserialize `out`, re-serialize, and require the
    /// same bytes. Serialization is deterministic, so byte equality is exactly
    /// field equality (and stricter than `==` for `-0.0` vs `0.0`).
    fn assert_round_trips(out: &HybridOutput) {
        let json = serde_json::to_string(out).expect("finite output serializes");
        let restored: HybridOutput = serde_json::from_str(&json).expect("round trip");
        let reserialized = serde_json::to_string(&restored).expect("finite restore serializes");
        assert_eq!(reserialized, json, "round trip must be lossless: {out:?}");
    }

    #[test]
    fn hybrid_output_finite_round_trips_both_shapes() {
        // Reverse-path shape: every optional field is `Some`, so a lossless
        // record carries no `null` at all.
        let reverse = reverse_shaped_output();
        let json = serde_json::to_string(&reverse).expect("finite output serializes");
        assert!(!json.contains("null"), "unexpected null in {json}");
        assert_round_trips(&reverse);

        // Tiny ANN→SNN shape: `None` is the legitimate wire spelling of null.
        let tiny = tiny_forward_output();
        let json = serde_json::to_string(&tiny).expect("finite output serializes");
        assert_eq!(
            json,
            "{\"embedding\":[0.125,-0.75,0.0],\"stimuli\":[0.5,-0.5],\
            \"fired_neurons\":[0],\"num_neurons\":2,\"global_step\":1,\"expert_weights\":null,\
            \"selected_experts\":null,\"routing_entropy\":null}"
        );
        assert_round_trips(&tiny);
    }

    /// Field mutation + expected error prefix, shared by the policy tests.
    type FieldCase<T> = (fn(&mut T), &'static str);

    #[test]
    fn hybrid_output_serialization_rejects_non_finite_per_field() {
        // (mutate, expected prefix) pairs: one per guarded f32 field, covering
        // every non-finite spelling at least once.
        let cases: Vec<FieldCase<HybridOutput>> = vec![
            (
                |o| o.embedding[1] = f32::NAN,
                "HybridOutput.embedding[1] is NaN",
            ),
            (
                |o| o.stimuli = vec![f32::INFINITY],
                "HybridOutput.stimuli[0] is +Inf",
            ),
            (
                |o| o.stimuli = vec![1.0, f32::NEG_INFINITY],
                "HybridOutput.stimuli[1] is -Inf",
            ),
            (
                |o| o.expert_weights = Some(vec![0.5, f32::NAN]),
                "HybridOutput.expert_weights[1] is NaN",
            ),
            (
                |o| o.routing_entropy = Some(f32::INFINITY),
                "HybridOutput.routing_entropy is +Inf",
            ),
        ];
        for (mutate, prefix) in cases {
            let mut out = reverse_shaped_output();
            mutate(&mut out);
            let err = serde_json::to_string(&out)
                .expect_err("non-finite output must fail closed, not write null");
            let text = err.to_string();
            assert!(text.starts_with(prefix), "expected {prefix}, got {text}");
            assert!(
                text.contains("JSON would emit null"),
                "policy reason in {text}"
            );
        }
    }

    #[test]
    fn hybrid_output_serialization_still_accepts_none_optionals() {
        let out = tiny_forward_output();
        let json = serde_json::to_string(&out).expect("`None` fields are finite-by-definition");
        assert!(json.contains("\"expert_weights\":null"));
        assert!(json.contains("\"routing_entropy\":null"));
    }

    #[test]
    fn hybrid_output_deserialization_rejects_null_inside_float_arrays() {
        // Legacy record shape produced by the pre-policy writer.
        let json = r#"{
            "embedding": [null],
            "stimuli": [0.5],
            "fired_neurons": [0],
            "num_neurons": 2,
            "global_step": 1,
            "expert_weights": null,
            "selected_experts": null,
            "routing_entropy": null
        }"#;
        let err =
            serde_json::from_str::<HybridOutput>(json).expect_err("a null element is not an f32");
        let text = err.to_string();
        assert!(
            text.contains("invalid type: null, expected f32"),
            "unexpected error: {text}"
        );
    }

    #[test]
    fn hybrid_output_legacy_null_entropy_decodes_as_none() {
        // Documented asymmetry: `null` is the wire spelling of `None`, so a
        // legacy `Some(NaN)`-turned-`null` is indistinguishable. Callers must
        // re-derive entropy rather than trust the value.
        let json = r#"{
            "embedding": [0.25],
            "stimuli": [],
            "fired_neurons": [],
            "num_neurons": 1,
            "global_step": 3,
            "expert_weights": null,
            "selected_experts": null,
            "routing_entropy": null
        }"#;
        let out: HybridOutput = serde_json::from_str(json).expect("null entropy decodes as None");
        assert_eq!(out.routing_entropy, None);
        assert_eq!(out.expert_weights, None);
    }
}
