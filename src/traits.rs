// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::capabilities::BackendCapabilities;
use crate::error::Result;
use crate::tensor::Tensor;
use crate::types::{Dtype, TensorManifestEntry, TensorRole};
use serde::{Deserialize, Serialize};

/// Transformer backend: token IDs → hidden-state tensor.
///
/// # Hidden-state contract
///
/// [`crate::HybridNetwork::forward`] validates the tensor returned by
/// [`hidden_states`](Self::hidden_states) **before** pooling, projection, or
/// [`SpikingNetwork::step`]. Failures are structured [`crate::HybridError`]
/// contract variants (not Sentry runtime failures).
///
/// Supported layouts:
///
/// - **Rank 2** `[seq_len, dim]` — per-token hidden states. `seq_len` must
///   equal `token_ids.len()` and `dim` must equal [`dim`](Self::dim). This is
///   the canonical shape.
/// - **Rank 1** `[dim]` — a pre-pooled embedding vector. The single axis must
///   equal [`dim`](Self::dim). Sequence length is not encoded in this layout.
///
/// Every other rank (0, 3, …) is rejected. Backing `data.len()` must equal the
/// layout product (`seq_len * dim` or `dim`). Every value must be finite
/// (no NaN or ±Inf).
pub trait Transformer {
    fn hidden_states(&self, token_ids: &[u32]) -> Tensor;
    fn dim(&self) -> usize;
    fn max_seq_len(&self) -> usize;
    fn param_count(&self) -> usize;

    /// Side-effect-free capability report.
    ///
    /// The default derives domain, dtype, hidden width, and sequence window
    /// from [`dim`](Self::dim) / [`max_seq_len`](Self::max_seq_len) and does
    /// not advertise batching, streaming, state, reset, or optional features.
    /// Backends that support those must override this method. Implementations
    /// must not run [`hidden_states`](Self::hidden_states) or mutate `self`.
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::ann(std::any::type_name::<Self>())
            .with_dtypes([Dtype::F32])
            .with_hidden_dim(self.dim())
            .with_max_sequence(self.max_seq_len())
    }
}

/// Spiking backend: bounded stimuli → fired neuron indices.
///
/// This is the **single ANN↔SNN step contract** for the stack (issue #61 /
/// RM-2205): the cortex-tensor `SnnBackend` surface was folded into it and
/// lives nowhere else. The boundary adapters are [`crate::SpikeEncoder`] (ANN
/// signal → stimuli) and [`crate::SpikeDecoder`] (fired IDs → readout); the
/// hybrid routing host composing all three is [`crate::SpikingExpertRouter`].
///
/// Input channels and output neurons are separate axes, stable over stepping.
/// [`crate::HybridNetwork::forward`] rejects either axis being zero before
/// projection or [`step`](Self::step), and checks returned fired-ID bounds.
pub trait SpikingNetwork {
    /// Consume exactly [`num_channels`](Self::num_channels) bounded stimuli.
    /// Return output neuron IDs in `0..num_neurons()`; do not reinterpret them
    /// as input-channel IDs or replicate spikes to fill an input-sized vector.
    fn step(&mut self, stimuli: &[f32], modulators: &NeuroModulators) -> Result<Vec<usize>>;

    /// Held-out tick: like [`step`](Self::step), but no plasticity-controlled
    /// state (weights, eligibility traces, thresholds) may change.
    ///
    /// The default **fails closed**, matching [`reset`](Self::reset): a backend
    /// that does not override it cannot be evaluated frozen, so a held-out tick
    /// can never silently run the learning path. A backend that overrides this
    /// method MUST also advertise the `frozen_evaluation`
    /// [`RequiredFeature`](crate::RequiredFeature) in
    /// [`capabilities`](Self::capabilities); hosts such as
    /// [`crate::SpikingExpertRouter::forward_frozen`] reject the call before
    /// stepping when the feature is absent.
    fn step_frozen(
        &mut self,
        _stimuli: &[f32],
        _modulators: &NeuroModulators,
    ) -> Result<Vec<usize>> {
        Err(crate::HybridError::SnnStep(
            "frozen evaluation is not supported by this backend".into(),
        ))
    }

    /// Stimulus input width, independent of output population.
    fn num_channels(&self) -> usize;

    /// Output population and fired-ID domain. Backends with unequal input and
    /// output widths MUST override this. The default preserves the existing
    /// one-output-neuron-per-input-channel contract for downstream implementors.
    fn num_neurons(&self) -> usize {
        self.num_channels()
    }

    /// Restore pre-step dynamics when the backend supports reset.
    ///
    /// Backends that advertise `BackendCapabilities::reset` must override this
    /// method. The default is an explicit unsupported-operation error.
    fn reset(&mut self) -> Result<()> {
        Err(crate::HybridError::SnnStep(
            "reset is not supported by this backend".into(),
        ))
    }

    /// Side-effect-free capability report.
    ///
    /// The default advertises the SNN domain, `f32`, the stimulus width, the
    /// output population, a one-step temporal window, and `step` (the trait
    /// method itself). It does **not**
    /// advertise cross-call state, reset, caller-controlled RNG, or
    /// neuromodulation — those are opt-in overrides. A stateless `step` that
    /// depends only on the current stimuli must not be treated as recurrent.
    /// Backends that override [`step_frozen`](Self::step_frozen) must add the
    /// `frozen_evaluation` feature here. Implementations must not call
    /// [`step`](Self::step).
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::snn(std::any::type_name::<Self>())
            .with_dtypes([Dtype::F32])
            .with_channels(self.num_channels())
            .with_num_neurons(self.num_neurons())
            .with_max_sequence(1)
            .with_features([crate::capabilities::RequiredFeature::new("step")])
    }
}

/// Neuromodulator state handed to every [`SpikingNetwork`] step.
///
/// # JSON / serde policy
///
/// This type is persisted in step records and telemetry, and `serde_json`
/// would silently write a non-finite `f32` as JSON `null`. Serialization
/// therefore fails closed: any `NaN` / ±`Inf` in a field returns an error
/// naming that field instead of writing `null` (crate `serde_finite` policy,
/// RM-1465).
/// Deserialization is unchanged — strict JSON cannot carry a non-finite number.
#[derive(Debug, Clone, Deserialize)]
pub struct NeuroModulators {
    pub dopamine: f32,
    pub cortisol: f32,
    pub acetylcholine: f32,
    pub tempo: f32,
    pub aux_dopamine: f32,
}

/// Fails closed on non-finite `f32` values instead of letting `serde_json`
/// write `null` (see [`NeuroModulators`] docs).
impl Serialize for NeuroModulators {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        crate::serde_finite::check_f32(self.dopamine, "NeuroModulators.dopamine")?;
        crate::serde_finite::check_f32(self.cortisol, "NeuroModulators.cortisol")?;
        crate::serde_finite::check_f32(self.acetylcholine, "NeuroModulators.acetylcholine")?;
        crate::serde_finite::check_f32(self.tempo, "NeuroModulators.tempo")?;
        crate::serde_finite::check_f32(self.aux_dopamine, "NeuroModulators.aux_dopamine")?;

        // Field order mirrors the struct declaration, as the derived impl did.
        let mut state = serializer.serialize_struct("NeuroModulators", 5)?;
        state.serialize_field("dopamine", &self.dopamine)?;
        state.serialize_field("cortisol", &self.cortisol)?;
        state.serialize_field("acetylcholine", &self.acetylcholine)?;
        state.serialize_field("tempo", &self.tempo)?;
        state.serialize_field("aux_dopamine", &self.aux_dopamine)?;
        state.end()
    }
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

/// Deprecated compatibility shim for a GGUF checkpoint layout.
///
/// Parsing, mmap, and layout discovery belong to `engram-parser`. Planning
/// input is [`AnnTopology`](crate::AnnTopology) or
/// [`HybridTopology`](crate::HybridTopology). See the migration guide in
/// `docs/implementing-backends.md`.
#[deprecated(
    since = "0.4.0",
    note = "parse checkpoints with engram-parser; plan from AnnTopology or HybridTopology"
)]
#[allow(deprecated)]
pub trait GgufLoader {
    fn load(&self, path: &str) -> Result<GgufLayout>;
}

/// Deprecated compatibility shim for a GGUF header summary.
///
/// Parsing belongs to `engram-parser`. Planning input is
/// [`AnnTopology`](crate::AnnTopology) or [`HybridTopology`](crate::HybridTopology).
#[deprecated(
    since = "0.4.0",
    note = "parse checkpoints with engram-parser; plan from AnnTopology or HybridTopology"
)]
#[allow(deprecated)]
pub struct GgufLayout {
    pub architecture: String,
    pub tensor_count: usize,
}

/// Safetensors / Hugging Face shard-index onboarding contract.
///
/// **Deprecated.** Extension point kept as a compatibility shim — **not**
/// called by [`HybridNetwork::forward`](crate::HybridNetwork::forward).
/// Concrete header parse, mmap, layout discovery, and payload extract belong
/// to `engram-parser`. Planning input is [`AnnTopology`](crate::AnnTopology)
/// or [`HybridTopology`](crate::HybridTopology).
///
/// Symmetric to [`GgufLoader`]: one method, path in, layout out. Reuse
/// [`HybridError::ModelLoad`](crate::HybridError::ModelLoad),
/// [`UnsupportedFormat`](crate::HybridError::UnsupportedFormat),
/// [`MissingTensor`](crate::HybridError::MissingTensor), and
/// [`SafetensorsParse`](crate::HybridError::SafetensorsParse) — do not add a
/// second competing Safetensors symbol.
#[deprecated(
    since = "0.4.0",
    note = "parse checkpoints with engram-parser; plan from AnnTopology or HybridTopology"
)]
#[allow(deprecated)]
pub trait SafetensorsLoader {
    /// Inventory a Safetensors file or Hugging Face shard index at `path`.
    ///
    /// `path` is a filesystem path to a `.safetensors` file or
    /// `model.safetensors.index.json`. This crate does not open it; backends
    /// in `engram-parser` perform header parse and mmap.
    ///
    /// Returns a header-only [`SafetensorsLayout`] (architecture, manifest,
    /// inferred roles). Typical errors:
    /// [`HybridError::SafetensorsParse`](crate::HybridError::SafetensorsParse)
    /// for header / shard-index failures,
    /// [`HybridError::ModelLoad`](crate::HybridError::ModelLoad) for I/O,
    /// [`HybridError::UnsupportedFormat`](crate::HybridError::UnsupportedFormat)
    /// when the file is not Safetensors, and
    /// [`HybridError::MissingTensor`](crate::HybridError::MissingTensor) when a
    /// required tensor is absent from the inventory.
    fn load(&self, path: &str) -> Result<SafetensorsLayout>;
}

/// Header-only Safetensors inventory (name, dtype, shape, shard refs, roles).
///
/// **Deprecated** compatibility shim. Richer than [`GgufLayout`] so existing
/// manifests keep their shape. `tensor_count` equals `tensors.len()` after
/// [`SafetensorsLayout::new`] and after deserialization (a mismatched
/// `tensor_count` in JSON is recomputed from `tensors`). Checkpoint opening
/// belongs to `engram-parser`; planning input is
/// [`HybridTopology`](crate::HybridTopology).
#[deprecated(
    since = "0.4.0",
    note = "parse checkpoints with engram-parser; plan from AnnTopology or HybridTopology"
)]
#[allow(deprecated)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "SafetensorsLayoutDe")]
pub struct SafetensorsLayout {
    /// Architecture tag from config / header metadata (empty if unknown).
    pub architecture: String,
    /// Number of tensors reported by the loader.
    pub tensor_count: usize,
    /// Ordered manifest. Roles are typically [`TensorRole::from_name`].
    pub tensors: Vec<TensorManifestEntry>,
}

#[allow(deprecated)]
impl SafetensorsLayout {
    /// Build a layout and set [`tensor_count`](Self::tensor_count) from the
    /// manifest length.
    pub fn new(architecture: String, tensors: Vec<TensorManifestEntry>) -> Self {
        let tensor_count = tensors.len();
        Self {
            architecture,
            tensor_count,
            tensors,
        }
    }

    /// Manifest entries whose [`role`](TensorManifestEntry::role) equals `role`,
    /// in inventory order. Pure filter — no I/O.
    pub fn entries_with_role(&self, role: TensorRole) -> Vec<&TensorManifestEntry> {
        self.tensors.iter().filter(|e| e.role == role).collect()
    }
}

/// Deserialize helper: ignore a caller-supplied `tensor_count` and rebuild
/// through [`SafetensorsLayout::new`] so the public count matches `tensors`.
#[derive(Deserialize)]
struct SafetensorsLayoutDe {
    architecture: String,
    /// Present in the serialized form so positional formats keep field order;
    /// [`SafetensorsLayout::new`] recomputes the public count from `tensors`.
    #[serde(default)]
    #[allow(dead_code)]
    tensor_count: usize,
    tensors: Vec<TensorManifestEntry>,
}

#[allow(deprecated)]
impl From<SafetensorsLayoutDe> for SafetensorsLayout {
    fn from(raw: SafetensorsLayoutDe) -> Self {
        Self::new(raw.architecture, raw.tensors)
    }
}

// ── Reverse path: SNN activity → (project) → MoE ExpertRouter ───────────────
// Shapes inspired by corinth-canal FunnelActivity / Router::forward / RouterOutput.
// No GIF dynamics, checkpoint load, or family adapters live here.

/// Pure spike / membrane surface for reverse-path projection (no neuron dynamics).
///
/// Field names mirror corinth-canal `FunnelActivity` / `Projector::project` inputs
/// (`spike_train`, `potentials`, `iz_potentials`) without ternary events or GIF state.
///
/// # JSON / serde policy
///
/// `potentials` and `iz_potentials` fail closed on non-finite values:
/// serialization returns an error naming the field instead of letting
/// `serde_json` write `null` (crate `serde_finite` policy, RM-1465). This
/// matches the runtime rule — [`crate::project_spike_activity`] already
/// rejects non-finite potentials — so a record written under this policy is
/// the only kind of trace that can round-trip.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SpikeActivity {
    /// Per-timestep fired neuron indices.
    pub spike_train: Vec<Vec<usize>>,
    /// Membrane potentials (typically length = neuron count).
    pub potentials: Vec<f32>,
    /// Optional adaptive-bank potentials; empty when unused.
    pub iz_potentials: Vec<f32>,
}

/// Fails closed on non-finite `f32` values instead of letting `serde_json`
/// write `null` (see [`SpikeActivity`] docs).
impl Serialize for SpikeActivity {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        crate::serde_finite::check_f32_slice(&self.potentials, "SpikeActivity.potentials")?;
        crate::serde_finite::check_f32_slice(&self.iz_potentials, "SpikeActivity.iz_potentials")?;

        // Field order mirrors the struct declaration, as the derived impl did.
        let mut state = serializer.serialize_struct("SpikeActivity", 3)?;
        state.serialize_field("spike_train", &self.spike_train)?;
        state.serialize_field("potentials", &self.potentials)?;
        state.serialize_field("iz_potentials", &self.iz_potentials)?;
        state.end()
    }
}

impl SpikeActivity {
    /// Build a one-step activity bag from fired indices (zeroed membranes).
    ///
    /// Pass [`SpikingNetwork::num_neurons`] (or [`crate::HybridOutput::num_neurons`]),
    /// NOT the input channel count. This is the reverse feature's neuron axis;
    /// a reduced population must not be padded to the input width. The projector
    /// cannot infer the true population from sparse fired IDs alone.
    ///
    /// # Errors
    ///
    /// - [`HybridError::InvalidConfig`] if `n_neurons == 0` (zero-extent neuron axis).
    /// - [`HybridError::InvalidConfig`] if any index in `fired` is `>= n_neurons`.
    ///
    /// An empty `fired` with `n_neurons > 0` is valid (silent step).
    pub fn from_fired(fired: &[usize], n_neurons: usize) -> Result<Self> {
        use crate::error::HybridError;
        if n_neurons == 0 {
            return Err(HybridError::InvalidConfig(
                "SpikeActivity::from_fired: n_neurons must be > 0".into(),
            ));
        }
        if let Some(&idx) = fired.iter().find(|&&i| i >= n_neurons) {
            return Err(HybridError::InvalidConfig(format!(
                "SpikeActivity::from_fired: fired index {idx} out of range for n_neurons={n_neurons}"
            )));
        }
        Ok(Self {
            spike_train: vec![fired.to_vec()],
            potentials: vec![0.0; n_neurons],
            iz_potentials: Vec::new(),
        })
    }
}

/// Result of MoE expert routing (corinth-canal `RouterOutput` subset + optional entropy).
///
/// On a successful [`ExpertRouter::route`] call, backends should satisfy:
/// - `expert_weights.len() == num_experts()`
/// - every weight is finite and `>= 0`
/// - weights sum to `1.0` within the tolerance below (normalized gate distribution)
/// - `selected_experts.len() == top_k()`
/// - selected indices are distinct and each `< num_experts()`
///
/// # Weight normalization tolerance
///
/// "Sums to approximately `1.0`" is an **enforced contract, not advice**.
/// [`ReverseHybridPath::forward_activity`](crate::ReverseHybridPath::forward_activity)
/// re-accumulates `expert_weights` in `f64` and requires the sum to be within
/// [`WEIGHT_SUM_TOLERANCE`](crate::WEIGHT_SUM_TOLERANCE) (`8 * f32::EPSILON`,
/// ~`9.54e-7`) of `1.0`. That bound is derived from the `f32` unit round-off and
/// is **independent of `num_experts()`**.
///
/// A sum inside the tolerance is renormalized in `f64` before it reaches
/// [`HybridOutput`](crate::HybridOutput). A sum outside it is **rejected** with
/// [`HybridError::InvalidConfig`](crate::HybridError::InvalidConfig); it is never
/// silently rescaled. Return a normalized distribution (e.g. via
/// [`softmax`](crate::softmax)), not raw gate scores: `f32` drift is tolerated, a
/// sum such as `0.9` or `2.0` is an error.
///
/// Accumulate your normalizer in `f64`. An `f32` denominator drifts by
/// `O(num_experts * 2^-24)` — up to ~`4.8e-2` at
/// [`MAX_REASONABLE_EXPERTS`](crate::MAX_REASONABLE_EXPERTS) — and will be
/// rejected at large expert counts.
///
/// # JSON / serde policy
///
/// `expert_weights` and `routing_entropy` fail closed on non-finite values:
/// serialization returns an error naming the field instead of letting
/// `serde_json` write `null` (crate `serde_finite` policy, RM-1465).
/// `routing_entropy` is passed through verbatim, so a router that reports
/// `Some(NaN)` is caught here even when every host-side weight check passes.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ExpertRouteOutput {
    /// Full gate distribution over experts (`len == num_experts`, sum ≈ 1).
    pub expert_weights: Vec<f32>,
    /// Selected expert indices (`len == top_k`, each `< num_experts`).
    pub selected_experts: Vec<usize>,
    /// Optional routing entropy for telemetry; `None` if not computed.
    pub routing_entropy: Option<f32>,
}

/// Fails closed on non-finite `f32` values instead of letting `serde_json`
/// write `null` (see [`ExpertRouteOutput`] docs).
impl Serialize for ExpertRouteOutput {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        crate::serde_finite::check_f32_slice(
            &self.expert_weights,
            "ExpertRouteOutput.expert_weights",
        )?;
        crate::serde_finite::check_opt_f32(
            self.routing_entropy,
            "ExpertRouteOutput.routing_entropy",
        )?;

        // Field order mirrors the struct declaration, as the derived impl did.
        let mut state = serializer.serialize_struct("ExpertRouteOutput", 3)?;
        state.serialize_field("expert_weights", &self.expert_weights)?;
        state.serialize_field("selected_experts", &self.selected_experts)?;
        state.serialize_field("routing_entropy", &self.routing_entropy)?;
        state.end()
    }
}

/// Mixture-of-Experts routing contract: embedding → expert weights + selection.
///
/// Concrete checkpoint-backed routers live outside this crate. Reference
/// implementations (e.g. uniform stub) may ship under the `backends` feature.
///
/// # Contract
///
/// - [`num_experts`](Self::num_experts) returns a value `> 0`.
/// - [`top_k`](Self::top_k) returns a value in `1..=num_experts()`.
/// - Embedding length is **not** fixed to a research constant (corinth uses 2048);
///   backends validate dimensions they care about. Empty embeddings must error.
/// - Successful [`route`](Self::route) results follow [`ExpertRouteOutput`] invariants
///   (normalized weights, valid top-k selection). Hosts enforce them: an
///   `expert_weights` sum outside
///   [`WEIGHT_SUM_TOLERANCE`](crate::WEIGHT_SUM_TOLERANCE) fails with
///   [`HybridError::InvalidConfig`](crate::HybridError::InvalidConfig) rather than
///   being silently renormalized.
pub trait ExpertRouter {
    /// Number of experts considered by this router (`> 0`).
    fn num_experts(&self) -> usize;
    /// Top-k selection size (`1..=num_experts()`).
    fn top_k(&self) -> usize;
    /// Route a dense embedding into expert weights and a top-k selection.
    ///
    /// Returns [`Err`] for empty embeddings or backend-specific validation failures.
    fn route(&mut self, embedding: &[f32]) -> Result<ExpertRouteOutput>;

    /// Pre-softmax gate logits for `embedding`: the raw, unnormalized scores
    /// whose softmax is [`ExpertRouteOutput::expert_weights`].
    ///
    /// The additive spike boost of [`crate::SpikingExpertRouter`] is defined on
    /// gate **logits**, not on the normalized weights [`route`](Self::route)
    /// returns (boosting a normalized weight cannot promote an expert
    /// monotonically — issue #93). That host therefore calls this method and
    /// re-normalizes with [`crate::routing::softmax`] itself.
    ///
    /// The default fails closed with [`crate::HybridError::InvalidConfig`] so a
    /// backend that only produces normalized weights is never silently
    /// misread: a router that cannot expose logits cannot be composed into
    /// [`crate::SpikingExpertRouter`]. Routers whose logits exist (stub and
    /// synthetic reference routers, checkpoint-backed gate matmuls) override
    /// this with the exact scores [`route`](Self::route) normalizes.
    ///
    /// Contract for overrides: returns exactly [`num_experts`](Self::num_experts)
    /// scores, rejects empty embeddings and backend-specific dimension
    /// mismatches, and leaves router state unchanged for a rejected call.
    /// `route` and this method must agree — same embedding, same scores.
    fn gate_logits(&mut self, embedding: &[f32]) -> Result<Vec<f32>> {
        let _ = embedding;
        Err(crate::HybridError::InvalidConfig(
            "this ExpertRouter does not expose pre-softmax gate logits; \
             SpikingExpertRouter requires them to apply the spike boost"
                .into(),
        ))
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::error::HybridError;
    use crate::types::{Dtype, TensorManifestEntry, TensorRole};

    fn sample_layout() -> SafetensorsLayout {
        SafetensorsLayout::new(
            "mixtral".into(),
            vec![
                TensorManifestEntry::new(
                    "model.layers.0.block_sparse_moe.gate.weight".into(),
                    Dtype::F16,
                    vec![8, 16],
                    Some("model-00001-of-00002.safetensors".into()),
                    vec![],
                ),
                TensorManifestEntry::new(
                    "model.layers.0.block_sparse_moe.experts.0.w1.weight".into(),
                    Dtype::BF16,
                    vec![16, 16],
                    Some("model-00002-of-00002.safetensors".into()),
                    vec!["expert:0".into()],
                ),
                TensorManifestEntry::new(
                    "model.layers.0.self_attn.q_proj.weight".into(),
                    Dtype::F32,
                    vec![16, 16],
                    None,
                    vec![],
                ),
            ],
        )
    }

    #[test]
    fn layout_tensor_count_matches_manifest_len() {
        let layout = sample_layout();
        assert_eq!(layout.tensor_count, 3);
        assert_eq!(layout.tensor_count, layout.tensors.len());
        assert_eq!(layout.architecture, "mixtral");
    }

    #[test]
    fn layout_entries_with_role_filters_manifest() {
        let layout = sample_layout();
        let routers = layout.entries_with_role(TensorRole::Router);
        assert_eq!(routers.len(), 1);
        assert_eq!(
            routers[0].name,
            "model.layers.0.block_sparse_moe.gate.weight"
        );
        let experts = layout.entries_with_role(TensorRole::ExpertWeight);
        assert_eq!(experts.len(), 1);
        assert_eq!(layout.entries_with_role(TensorRole::Norm).len(), 0);
    }

    #[test]
    fn layout_serde_round_trip() {
        let layout = sample_layout();
        let json = serde_json::to_string(&layout).unwrap();
        let back: SafetensorsLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(back, layout);
    }

    #[test]
    fn layout_deserialize_recomputes_mismatched_tensor_count() {
        let json = r#"{
            "architecture": "mixtral",
            "tensor_count": 0,
            "tensors": [{
                "name": "model.layers.0.mlp.gate.weight",
                "dtype": "F16",
                "shape": [8, 16],
                "shard": null,
                "role": "Router",
                "labels": []
            }]
        }"#;
        let layout: SafetensorsLayout = serde_json::from_str(json).unwrap();
        assert_eq!(layout.tensor_count, 1);
        assert_eq!(layout.tensor_count, layout.tensors.len());
        assert_eq!(layout.tensors[0].name, "model.layers.0.mlp.gate.weight");
    }

    struct MemorySafetensorsLoader {
        layout: SafetensorsLayout,
    }

    impl SafetensorsLoader for MemorySafetensorsLoader {
        fn load(&self, path: &str) -> Result<SafetensorsLayout> {
            if path.is_empty() {
                return Err(HybridError::SafetensorsParse(
                    "empty path is not a Safetensors file".into(),
                ));
            }
            Ok(self.layout.clone())
        }
    }

    #[test]
    fn loader_trait_returns_layout_and_maps_parse_errors() {
        let loader = MemorySafetensorsLoader {
            layout: sample_layout(),
        };
        let layout = loader.load("model.safetensors").unwrap();
        assert_eq!(layout.tensor_count, 3);
        let err = loader.load("").unwrap_err();
        assert!(matches!(err, HybridError::SafetensorsParse(_)));
        assert!(err.to_string().contains("Safetensors parse failed"));
    }

    // ── Non-finite JSON policy (RM-1465) ────────────────────────────────────

    #[test]
    fn neuro_modulators_finite_round_trip() {
        let mods = NeuroModulators {
            dopamine: 0.25,
            cortisol: 0.75,
            acetylcholine: 1.5,
            tempo: 2.0,
            aux_dopamine: -0.5,
        };
        let json = serde_json::to_string(&mods).expect("finite modulators serialize");
        assert!(!json.contains("null"), "unexpected null in {json}");
        let restored: NeuroModulators = serde_json::from_str(&json).expect("round trip");
        assert_eq!(restored.dopamine, mods.dopamine);
        assert_eq!(restored.cortisol, mods.cortisol);
        assert_eq!(restored.acetylcholine, mods.acetylcholine);
        assert_eq!(restored.tempo, mods.tempo);
        assert_eq!(restored.aux_dopamine, mods.aux_dopamine);
    }

    /// Field mutation + expected error prefix, shared by the policy tests.
    type FieldCase<T> = (fn(&mut T), &'static str);

    #[test]
    fn neuro_modulators_serialization_rejects_non_finite_per_field() {
        let cases: Vec<FieldCase<NeuroModulators>> = vec![
            (|m| m.dopamine = f32::NAN, "NeuroModulators.dopamine is NaN"),
            (
                |m| m.cortisol = f32::INFINITY,
                "NeuroModulators.cortisol is +Inf",
            ),
            (
                |m| m.acetylcholine = f32::NEG_INFINITY,
                "NeuroModulators.acetylcholine is -Inf",
            ),
            (|m| m.tempo = f32::NAN, "NeuroModulators.tempo is NaN"),
            (
                |m| m.aux_dopamine = f32::INFINITY,
                "NeuroModulators.aux_dopamine is +Inf",
            ),
        ];
        for (mutate, prefix) in cases {
            let mut mods = NeuroModulators::default();
            mutate(&mut mods);
            let err = serde_json::to_string(&mods)
                .expect_err("non-finite modulators must fail closed, not write null");
            let text = err.to_string();
            assert!(text.starts_with(prefix), "expected {prefix}, got {text}");
            assert!(
                text.contains("JSON would emit null"),
                "policy reason in {text}"
            );
        }
    }

    #[test]
    fn spike_activity_serialization_rejects_non_finite_potentials() {
        let cases: Vec<FieldCase<SpikeActivity>> = vec![
            (
                |a| a.potentials[1] = f32::NAN,
                "SpikeActivity.potentials[1] is NaN",
            ),
            (
                |a| a.iz_potentials[0] = f32::INFINITY,
                "SpikeActivity.iz_potentials[0] is +Inf",
            ),
        ];
        for (mutate, prefix) in cases {
            let mut act = SpikeActivity {
                spike_train: vec![vec![0]],
                potentials: vec![0.0, 1.0],
                iz_potentials: vec![0.0, 2.0],
            };
            mutate(&mut act);
            let err = serde_json::to_string(&act)
                .expect_err("non-finite potentials must fail closed, not write null");
            let text = err.to_string();
            assert!(text.starts_with(prefix), "expected {prefix}, got {text}");
            assert!(
                text.contains("JSON would emit null"),
                "policy reason in {text}"
            );
        }
    }

    #[test]
    fn spike_activity_finite_round_trip() {
        let act = SpikeActivity {
            spike_train: vec![vec![0, 2], vec![1]],
            potentials: vec![-0.5, 0.0, 1.25],
            iz_potentials: vec![0.75],
        };
        let json = serde_json::to_string(&act).expect("finite activity serializes");
        let restored: SpikeActivity = serde_json::from_str(&json).expect("round trip");
        assert_eq!(restored, act);
    }

    #[test]
    fn expert_route_output_rejects_non_finite_weights_and_entropy() {
        let cases: Vec<FieldCase<ExpertRouteOutput>> = vec![
            (
                |r| r.expert_weights[1] = f32::NAN,
                "ExpertRouteOutput.expert_weights[1] is NaN",
            ),
            (
                |r| r.routing_entropy = Some(f32::INFINITY),
                "ExpertRouteOutput.routing_entropy is +Inf",
            ),
        ];
        for (mutate, prefix) in cases {
            let mut route = ExpertRouteOutput {
                expert_weights: vec![0.25, 0.75],
                selected_experts: vec![1],
                routing_entropy: Some(0.42),
            };
            mutate(&mut route);
            let err = serde_json::to_string(&route)
                .expect_err("non-finite route must fail closed, not write null");
            let text = err.to_string();
            assert!(text.starts_with(prefix), "expected {prefix}, got {text}");
            assert!(
                text.contains("JSON would emit null"),
                "policy reason in {text}"
            );
        }
    }

    #[test]
    fn expert_route_output_finite_round_trip() {
        let route = ExpertRouteOutput {
            expert_weights: vec![0.25, 0.75],
            selected_experts: vec![1],
            routing_entropy: Some(0.42),
        };
        let json = serde_json::to_string(&route).expect("finite route serializes");
        assert!(!json.contains("null"), "unexpected null in {json}");
        let restored: ExpertRouteOutput = serde_json::from_str(&json).expect("round trip");
        assert_eq!(restored, route);

        let none_entropy = ExpertRouteOutput {
            expert_weights: vec![1.0],
            selected_experts: vec![0],
            routing_entropy: None,
        };
        let json = serde_json::to_string(&none_entropy).expect("`None` entropy serializes");
        assert!(json.contains("\"routing_entropy\":null"));
        let restored: ExpertRouteOutput = serde_json::from_str(&json).expect("round trip");
        assert_eq!(restored, none_entropy);
    }
}
