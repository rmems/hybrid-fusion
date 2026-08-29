// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::Result;
use crate::tensor::Tensor;
use crate::types::PrecisionTier;
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
    /// Build a one-step activity bag from fired indices (zeroed membranes).
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertRouteOutput {
    /// Full gate distribution over experts (`len == num_experts`, sum ≈ 1).
    pub expert_weights: Vec<f32>,
    /// Selected expert indices (`len == top_k`, each `< num_experts`).
    pub selected_experts: Vec<usize>,
    /// Optional routing entropy for telemetry; `None` if not computed.
    pub routing_entropy: Option<f32>,
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
}

// ── Dry-run precision planning: stage names → planned ops (no weights) ─────
// Reduced from grok-ozempic `DryRunPlanner` / `PlannedKernelCall`. No manifest
// loader, coverage arithmetic, GIF thresholds, or `BackendKernel` names here —
// see docs/extraction-map.md section D.

/// What a backend would do to realize a [`PlannedOperation`]'s tier.
///
/// Deliberately **not** a kernel name: `BackendKernel` and its methods belong to
/// `myelin-accelerator`, not to this crate. These two variants carry the only
/// orchestration-level bit the research planner's four `kernel_method` strings
/// encoded — whether the stage must be transformed or is already conformant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationKind {
    /// Backend must transform the stage into its tier; serializes as `"convert"`.
    Convert,
    /// Stage already sits at its tier; the backend wraps it with no math.
    /// Serializes as `"passthrough"`.
    Passthrough,
}

/// One planned unit of work from a dry run: a stage name, its tier, and what a
/// backend would do with it.
///
/// Reduced from grok-ozempic `PlannedKernelCall`. Its `class`, `gif_threshold`,
/// `estimated_tensor_count`, and `kernel_method` fields are all outside this
/// crate's surface — see [`HybridStagePlanner`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlannedOperation {
    /// Stage name this entry plans, echoed verbatim from the planner's input.
    pub stage: String,
    /// Precision tier the stage is planned at.
    pub tier: PrecisionTier,
    /// Whether realizing `tier` needs a conversion or a pass-through wrap.
    pub kind: OperationKind,
}

/// Dry-run planning contract: pipeline / tensor stage **names** → planned
/// operations, with nothing loaded.
///
/// Concrete planners — manifest rule sets, model inventories, coverage audits —
/// live outside this crate; hybrid-fusion owns the vocabulary and the contract
/// only. There is no reference implementation under `backends`: a dry run is a
/// policy decision, not math.
///
/// # Contract
///
/// - Planning is **name-only**. Implementations must not open, mmap, stat, or
///   read weight files, manifests, or checkpoints. A stage name is a structural
///   tensor / stage identifier, never a filesystem path. This is what makes the
///   run "dry", and why the input carries no shape, dtype, or byte count.
/// - [`plan`](Self::plan) returns exactly one [`PlannedOperation`] per input
///   stage, in input order: `out.len() == stages.len()` and
///   `out[i].stage == stages[i]`.
/// - A stage no rule matches is planned at [`default_tier`](Self::default_tier).
/// - `plan` is deterministic: the same `&self` and the same `stages` produce an
///   equal plan.
/// - [`PrecisionTier`] and [`OperationKind`] are independent axes — a
///   [`Preserve`](PrecisionTier::Preserve) stage may still need a
///   [`Convert`](OperationKind::Convert).
pub trait HybridStagePlanner {
    /// Tier applied to stages no rule matches (`ternary_snn` in the research pipeline).
    fn default_tier(&self) -> PrecisionTier;

    /// Plan every stage name into a [`PlannedOperation`], loading nothing.
    ///
    /// # Errors
    ///
    /// - [`HybridError::InvalidConfig`](crate::HybridError::InvalidConfig) if any
    ///   `stages` entry is empty — a zero-extent input, rejected the way a
    ///   zero-length tensor axis is.
    /// - [`HybridError::InvalidConfig`](crate::HybridError::InvalidConfig) for
    ///   planner-specific rule conflicts.
    ///
    /// An empty `stages` slice is valid and yields an empty plan.
    fn plan(&self, stages: &[&str]) -> Result<Vec<PlannedOperation>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::HybridError;

    /// Deterministic mock planner (no `backends` feature needed).
    ///
    /// Rules are name-substring only — the contract forbids consulting anything
    /// else — and mirror the research pipeline's MoE intent: routing tensors are
    /// preserved, experts arrive already ternary, attention is fp16.
    #[derive(Debug)]
    struct MockStagePlanner {
        default_tier: PrecisionTier,
    }

    impl MockStagePlanner {
        fn new(default_tier: PrecisionTier) -> Self {
            Self { default_tier }
        }
    }

    impl HybridStagePlanner for MockStagePlanner {
        fn default_tier(&self) -> PrecisionTier {
            self.default_tier
        }

        fn plan(&self, stages: &[&str]) -> Result<Vec<PlannedOperation>> {
            let mut plan = Vec::with_capacity(stages.len());
            for &stage in stages {
                if stage.is_empty() {
                    return Err(HybridError::InvalidConfig(
                        "MockStagePlanner::plan: stage name must not be empty".into(),
                    ));
                }
                // Preserve still converts (the research path writes f16 bytes):
                // tier and kind are independent axes.
                let (tier, kind) = if stage.contains("gate") || stage.contains("router") {
                    (PrecisionTier::Preserve, OperationKind::Convert)
                } else if stage.contains("expert") {
                    (PrecisionTier::TernarySnn, OperationKind::Passthrough)
                } else if stage.contains("attn") {
                    (PrecisionTier::Fp16, OperationKind::Convert)
                } else {
                    (self.default_tier, OperationKind::Convert)
                };
                plan.push(PlannedOperation {
                    stage: stage.to_string(),
                    tier,
                    kind,
                });
            }
            Ok(plan)
        }
    }

    #[test]
    fn plan_returns_one_operation_per_stage_in_order() {
        let planner = MockStagePlanner::new(PrecisionTier::TernarySnn);
        let stages = [
            "blk.0.moe_gate",
            "blk.0.expert.0",
            "blk.0.attn_q",
            "tok_embd",
        ];
        let plan = planner.plan(&stages).unwrap();
        assert_eq!(plan.len(), stages.len());
        for (op, name) in plan.iter().zip(stages) {
            assert_eq!(op.stage, name);
        }
    }

    /// An empty pipeline is valid, not an error — the same posture as an empty
    /// `fired` slice in [`SpikeActivity::from_fired`].
    #[test]
    fn plan_of_empty_slice_is_empty_plan() {
        let planner = MockStagePlanner::new(PrecisionTier::Preserve);
        assert!(planner.plan(&[]).unwrap().is_empty());
    }

    /// Zero-extent input: an empty stage name is rejected like a zero-length
    /// tensor axis, and it fails the whole plan rather than being skipped.
    #[test]
    fn plan_rejects_empty_stage_name() {
        let planner = MockStagePlanner::new(PrecisionTier::Fp16);
        match planner.plan(&["blk.0.attn_q", ""]).unwrap_err() {
            HybridError::InvalidConfig(msg) => assert!(msg.contains("stage name"), "{msg}"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    /// grok-ozempic resolves `TensorClass::Default` by reading
    /// `manifest.defaults.precision`. That manifest is out of this crate, so the
    /// fallback is declared by the implementation instead.
    #[test]
    fn unmatched_stage_falls_back_to_default_tier() {
        for tier in [
            PrecisionTier::Preserve,
            PrecisionTier::Fp16,
            PrecisionTier::TernarySnn,
        ] {
            let planner = MockStagePlanner::new(tier);
            let plan = planner.plan(&["blk.0.unknown_thing"]).unwrap();
            assert_eq!(plan[0].tier, planner.default_tier());
            assert_eq!(plan[0].tier, tier);
        }
    }

    /// MoE-awareness is the point of the tier vocabulary: routing tensors plan at
    /// `preserve`, never at `ternary_snn`, whatever the default is.
    #[test]
    fn moe_routing_stages_plan_at_preserve() {
        let planner = MockStagePlanner::new(PrecisionTier::TernarySnn);
        let plan = planner
            .plan(&["blk.0.moe_gate", "blk.1.expert_router"])
            .unwrap();
        assert!(plan.iter().all(|op| op.tier == PrecisionTier::Preserve));
    }

    /// Tier and kind are independent axes: an already-ternary expert is a
    /// pass-through, while a preserved tensor still needs a conversion. This is
    /// the whole content of the research planner's `kernel_method` strings, minus
    /// the kernel names.
    #[test]
    fn tier_and_kind_are_independent() {
        let planner = MockStagePlanner::new(PrecisionTier::TernarySnn);
        let plan = planner.plan(&["blk.0.expert.0", "blk.0.moe_gate"]).unwrap();
        assert_eq!(plan[0].tier, PrecisionTier::TernarySnn);
        assert_eq!(plan[0].kind, OperationKind::Passthrough);
        assert_eq!(plan[1].tier, PrecisionTier::Preserve);
        assert_eq!(plan[1].kind, OperationKind::Convert);
    }

    #[test]
    fn plan_is_deterministic() {
        let planner = MockStagePlanner::new(PrecisionTier::Fp16);
        let stages = ["blk.0.attn_q", "blk.0.expert.3", "x"];
        assert_eq!(
            planner.plan(&stages).unwrap(),
            planner.plan(&stages).unwrap()
        );
    }

    #[test]
    fn planned_operation_round_trips_through_json() {
        let op = PlannedOperation {
            stage: "blk.0.expert.0".into(),
            tier: PrecisionTier::TernarySnn,
            kind: OperationKind::Passthrough,
        };
        let json = serde_json::to_string(&op).unwrap();
        assert!(json.contains("\"ternary_snn\""), "{json}");
        assert!(json.contains("\"passthrough\""), "{json}");
        assert_eq!(serde_json::from_str::<PlannedOperation>(&json).unwrap(), op);
    }
}
