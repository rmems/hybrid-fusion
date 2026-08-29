// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reverse-path host: SNN activity → embedding → MoE route.
//!
//! Extracted from corinth-canal `Model` (`projector` + `router` half of
//! `forward_activity`). Does not own neuron dynamics or checkpoint I/O.
//! Dual host alongside [`crate::HybridNetwork`] (ANN → SNN).

use crate::error::{HybridError, Result};
use crate::projector::project_spike_activity;
use crate::traits::{ExpertRouter, SpikeActivity};
use crate::types::{HybridOutput, ProjectionMode};

/// Maximum accepted deviation of a router's `expert_weights` sum from `1.0`.
///
/// # Derivation (not a tuned constant)
///
/// A gate distribution emitted as `f32` cannot re-accumulate to exactly `1.0`:
/// each weight carries at most one `f32` rounding, so re-summing them in `f64`
/// differs from the router's own normalizer by at most
///
/// ```text
/// sum_i |w_i| * u  =  1 * u  =  2^-24  ~=  5.96e-8      (u = f32::EPSILON / 2)
/// ```
///
/// plus at most `n * 2^-53 ~= 1.1e-10` for re-accumulating
/// `n <= MAX_REASONABLE_EXPERTS` terms in `f64`. Total: `< 5.97e-8`.
///
/// **The bound does not depend on the expert count.** It is a property of the
/// `f32` weight format, not of `n` -- provided the router builds its normalizer
/// in `f64`, as [`crate::routing::softmax`] does. A router that accumulates an
/// `f32` denominator instead drifts by `O(n * u)` (measured up to `4.8e-2` at
/// [`MAX_REASONABLE_EXPERTS`](crate::routing::MAX_REASONABLE_EXPERTS)), which is
/// larger than any error a caller can absorb; that is a bug to fix in the
/// router, not a tolerance to widen.
///
/// `8 * f32::EPSILON` leaves ~16x headroom over the `5.97e-8` bound while still
/// rejecting every materially unnormalized distribution (e.g. `[2.0, 0.0]`, or a
/// sum of `0.991`) by three or more orders of magnitude.
pub const WEIGHT_SUM_TOLERANCE: f64 = 8.0 * f32::EPSILON as f64; // ~= 9.54e-7

/// Reverse-path host: SNN activity → embedding → MoE route.
///
/// Extracted from corinth-canal `Model` (`projector` + `router` half of
/// `forward_activity`). Does not own neuron dynamics or checkpoint I/O.
#[derive(Debug)]
pub struct ReverseHybridPath<R: ExpertRouter> {
    mode: ProjectionMode,
    n_neurons: usize,
    embed_dim: usize,
    router: R,
    global_step: u64,
}

impl<R: ExpertRouter> ReverseHybridPath<R> {
    /// Rejects `n_neurons == 0` or `embed_dim == 0`.
    pub fn new(
        mode: ProjectionMode,
        n_neurons: usize,
        embed_dim: usize,
        router: R,
    ) -> Result<Self> {
        if n_neurons == 0 {
            return Err(HybridError::InvalidConfig(
                "ReverseHybridPath: n_neurons must be > 0".into(),
            ));
        }
        if embed_dim == 0 {
            return Err(HybridError::InvalidConfig(
                "ReverseHybridPath: embed_dim must be > 0".into(),
            ));
        }
        Ok(Self {
            mode,
            n_neurons,
            embed_dim,
            router,
            global_step: 0,
        })
    }

    /// Projection mode used for spike-to-embedding conversion.
    pub fn mode(&self) -> ProjectionMode {
        self.mode
    }

    /// Number of neurons the reverse path was configured for.
    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }

    /// Embedding dimensionality produced by the projector.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Immutable reference to the configured router.
    pub fn router(&self) -> &R {
        &self.router
    }

    /// Mutable reference to the configured router.
    pub fn router_mut(&mut self) -> &mut R {
        &mut self.router
    }

    /// Current global step counter.
    pub fn global_step(&self) -> u64 {
        self.global_step
    }

    /// Reset the global step counter to zero.
    pub fn reset(&mut self) {
        self.global_step = 0;
    }

    /// Project activity, route through MoE, return `HybridOutput` with MoE fields set.
    ///
    /// Semantics match corinth-canal `Model::forward_activity` (projector + router half):
    /// 1. `embedding = project_spike_activity(...)`
    /// 2. `route = router.route(&embedding)` — **no Sentry capture** on reverse v1
    /// 3. Validate [`ExpertRouteOutput`](crate::ExpertRouteOutput) invariants,
    ///    including that `expert_weights` sums to `1.0` within
    ///    [`WEIGHT_SUM_TOLERANCE`]; accepted weights are renormalized in `f64`,
    ///    a sum outside the tolerance is an
    ///    [`InvalidConfig`](crate::HybridError::InvalidConfig) error rather than
    ///    a silent rescale
    /// 4. `global_step = saturating_add(1)` only after projection/routing succeed
    /// 5. Build `HybridOutput` (empty `stimuli`; `fired_neurons` = last non-empty spike step)
    pub fn forward_activity(&mut self, activity: &SpikeActivity) -> Result<HybridOutput> {
        let embedding =
            project_spike_activity(self.mode, activity, self.n_neurons, self.embed_dim)?;

        // v1: propagate router errors without Sentry (validation-heavy path;
        // avoid flooding on bad activity / empty embeddings).
        let route = self.router.route(&embedding)?;

        // Spike indices are already validated by `spike_activity_features` inside
        // `project_spike_activity`; `last_fired` is only called after that succeeds.
        let fired_neurons = last_fired(&activity.spike_train);

        // Defensive validation of the ExpertRouter contract from `src/traits.rs`.
        let n_experts = self.router.num_experts();
        let top_k = self.router.top_k();
        if n_experts == 0 || top_k == 0 || top_k > n_experts {
            return Err(HybridError::InvalidConfig(format!(
                "ReverseHybridPath: router reports invalid num_experts={n_experts} / top_k={top_k}"
            )));
        }
        if route.expert_weights.len() != n_experts {
            return Err(HybridError::InvalidConfig(format!(
                "ReverseHybridPath: expert_weights.len() ({}) != num_experts ({n_experts})",
                route.expert_weights.len()
            )));
        }
        if !route
            .expert_weights
            .iter()
            .all(|&w| w.is_finite() && w >= 0.0)
        {
            return Err(HybridError::InvalidConfig(
                "ReverseHybridPath: expert_weights contain non-finite or negative values".into(),
            ));
        }
        // A conforming router returns weights that re-accumulate to 1.0 within
        // WEIGHT_SUM_TOLERANCE regardless of expert count (see its derivation).
        // Accepted weights are still renormalized in f64 so the emitted vector
        // is exact wherever in the band the router landed.
        let weights_sum: f64 = route.expert_weights.iter().map(|&w| f64::from(w)).sum();
        // Every weight is already known finite and >= 0, so the f64 sum can be
        // neither non-finite nor negative (even usize::MAX terms of f32::MAX sum
        // to 6.3e57, far inside f64 range). The one degenerate case left is an
        // all-zero distribution; name it rather than letting it fall through to
        // the tolerance message below.
        if weights_sum == 0.0 {
            return Err(HybridError::InvalidConfig(
                "ReverseHybridPath: expert_weights are all zero".into(),
            ));
        }
        if (weights_sum - 1.0).abs() > WEIGHT_SUM_TOLERANCE {
            return Err(HybridError::InvalidConfig(format!(
                "ReverseHybridPath: expert_weights sum {weights_sum} is outside tolerance \
                 {WEIGHT_SUM_TOLERANCE} of 1.0; routers with many experts must accumulate \
                 their softmax denominator in f64 (see ExpertRouteOutput docs)"
            )));
        }
        // No post-check on the renormalized sum: by the bound above it is within
        // ~5.97e-8 of 1.0 by construction, so any threshold loose enough to be
        // meaningful would be unreachable.
        let scale = 1.0 / weights_sum;
        let expert_weights: Vec<f32> = route
            .expert_weights
            .iter()
            .map(|&w| (f64::from(w) * scale) as f32)
            .collect();
        if route.selected_experts.len() != top_k {
            return Err(HybridError::InvalidConfig(format!(
                "ReverseHybridPath: selected_experts.len() ({}) != top_k ({top_k})",
                route.selected_experts.len()
            )));
        }
        let mut seen = std::collections::HashSet::new();
        for &idx in &route.selected_experts {
            if idx >= n_experts {
                return Err(HybridError::InvalidConfig(format!(
                    "ReverseHybridPath: selected expert index {idx} >= num_experts ({n_experts})"
                )));
            }
            if !seen.insert(idx) {
                return Err(HybridError::InvalidConfig(
                    "ReverseHybridPath: selected_experts contains duplicate indices".into(),
                ));
            }
        }

        self.global_step = self.global_step.saturating_add(1);

        Ok(HybridOutput {
            embedding,
            stimuli: Vec::new(),
            fired_neurons,
            global_step: self.global_step,
            expert_weights: Some(expert_weights),
            selected_experts: Some(route.selected_experts),
            routing_entropy: route.routing_entropy,
        })
    }
}

/// Last non-empty step of the spike train; empty if none fired.
fn last_fired(spike_train: &[Vec<usize>]) -> Vec<usize> {
    spike_train
        .iter()
        .rev()
        .find(|step| !step.is_empty())
        .cloned()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ExpertRouteOutput;

    /// Deterministic mock router for unit tests (no `backends` feature).
    #[derive(Debug)]
    struct MockRouter {
        num_experts: usize,
        top_k: usize,
    }

    impl MockRouter {
        fn new(num_experts: usize, top_k: usize) -> Self {
            Self {
                num_experts,
                // `.max(1)` keeps `1.0 / k` finite; matches tests/reverse_path.rs.
                top_k: top_k.min(num_experts).max(1),
            }
        }
    }

    impl ExpertRouter for MockRouter {
        fn num_experts(&self) -> usize {
            self.num_experts
        }

        fn top_k(&self) -> usize {
            self.top_k
        }

        fn route(&mut self, embedding: &[f32]) -> Result<ExpertRouteOutput> {
            if embedding.is_empty() {
                return Err(HybridError::InvalidConfig(
                    "MockRouter: empty embedding".into(),
                ));
            }
            let n = self.num_experts;
            let mut expert_weights = vec![0.0f32; n];
            let k = self.top_k;
            let w = 1.0 / k as f32;
            let mut selected = Vec::with_capacity(k);
            for (i, weight) in expert_weights.iter_mut().enumerate().take(k) {
                *weight = w;
                selected.push(i);
            }
            Ok(ExpertRouteOutput {
                expert_weights,
                selected_experts: selected,
                routing_entropy: Some(0.5),
            })
        }
    }

    /// Router returning a caller-supplied weight vector verbatim, so tests can
    /// pin the [`WEIGHT_SUM_TOLERANCE`] boundary. `selected_experts` is the first
    /// `top_k` indices, which always satisfies the selection checks that follow.
    #[derive(Debug)]
    struct FixedWeightsRouter {
        weights: Vec<f32>,
        top_k: usize,
    }

    impl FixedWeightsRouter {
        fn new(weights: Vec<f32>, top_k: usize) -> Self {
            Self { weights, top_k }
        }
    }

    impl ExpertRouter for FixedWeightsRouter {
        fn num_experts(&self) -> usize {
            self.weights.len()
        }

        fn top_k(&self) -> usize {
            self.top_k
        }

        fn route(&mut self, _embedding: &[f32]) -> Result<ExpertRouteOutput> {
            Ok(ExpertRouteOutput {
                expert_weights: self.weights.clone(),
                selected_experts: (0..self.top_k).collect(),
                routing_entropy: Some(0.25),
            })
        }
    }

    /// Router that advertises `top_k() == 0` to test the explicit zero guard.
    #[derive(Debug)]
    struct ZeroTopKRouter;

    impl ExpertRouter for ZeroTopKRouter {
        fn num_experts(&self) -> usize {
            4
        }

        fn top_k(&self) -> usize {
            0
        }

        fn route(&mut self, _embedding: &[f32]) -> Result<ExpertRouteOutput> {
            Ok(ExpertRouteOutput {
                expert_weights: vec![0.25_f32; 4],
                selected_experts: vec![],
                routing_entropy: None,
            })
        }
    }

    #[test]
    fn forward_activity_rejects_zero_top_k() {
        let mut path =
            ReverseHybridPath::new(ProjectionMode::RateSum, 4, 8, ZeroTopKRouter).unwrap();
        let act = SpikeActivity::from_fired(&[0], 4).unwrap();
        let err = path.forward_activity(&act).unwrap_err();
        match err {
            HybridError::InvalidConfig(msg) => assert!(msg.contains("top_k=0")),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn new_rejects_zero_n_neurons() {
        let r = MockRouter::new(4, 2);
        let err = ReverseHybridPath::new(ProjectionMode::RateSum, 0, 8, r).unwrap_err();
        match err {
            HybridError::InvalidConfig(msg) => assert!(msg.contains("n_neurons")),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn new_rejects_zero_embed_dim() {
        let r = MockRouter::new(4, 2);
        let err = ReverseHybridPath::new(ProjectionMode::RateSum, 4, 0, r).unwrap_err();
        match err {
            HybridError::InvalidConfig(msg) => assert!(msg.contains("embed_dim")),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn forward_activity_increments_global_step() {
        let r = MockRouter::new(4, 2);
        let mut path = ReverseHybridPath::new(ProjectionMode::RateSum, 4, 8, r).unwrap();
        assert_eq!(path.global_step(), 0);
        let act = SpikeActivity::from_fired(&[0, 2], 4).unwrap();
        let out = path.forward_activity(&act).unwrap();
        assert_eq!(out.global_step, 1);
        assert_eq!(path.global_step(), 1);
        let out2 = path.forward_activity(&act).unwrap();
        assert_eq!(out2.global_step, 2);
    }

    #[test]
    fn reset_clears_global_step() {
        let r = MockRouter::new(4, 2);
        let mut path = ReverseHybridPath::new(ProjectionMode::RateSum, 4, 8, r).unwrap();
        let act = SpikeActivity::from_fired(&[1], 4).unwrap();
        path.forward_activity(&act).unwrap();
        path.reset();
        assert_eq!(path.global_step(), 0);
    }

    #[test]
    fn forward_activity_sets_moe_fields_and_empty_stimuli() {
        let r = MockRouter::new(4, 2);
        let mut path = ReverseHybridPath::new(ProjectionMode::RateSum, 4, 8, r).unwrap();
        let act = SpikeActivity::from_fired(&[0, 3], 4).unwrap();
        let out = path.forward_activity(&act).unwrap();

        assert_eq!(out.embedding.len(), 8);
        assert!(out.stimuli.is_empty());
        assert_eq!(out.fired_neurons, vec![0, 3]);
        let weights = out.expert_weights.expect("MoE weights");
        assert_eq!(weights.len(), 4);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        let selected = out.selected_experts.expect("selected");
        assert_eq!(selected.len(), 2);
        assert!(out.routing_entropy.is_some());
    }

    /// Drives the rejection branch from both directions and pins the documented
    /// step semantics: a rejected route must not consume a `global_step`.
    #[test]
    fn forward_activity_rejects_weights_outside_tolerance() {
        let act = SpikeActivity::from_fired(&[0], 4).unwrap();

        for (weights, label) in [
            (vec![2.0_f32, 0.0], "sum far above 1.0"),
            (vec![0.25_f32, 0.25], "sum far below 1.0"),
        ] {
            let mut path = ReverseHybridPath::new(
                ProjectionMode::RateSum,
                4,
                8,
                FixedWeightsRouter::new(weights, 1),
            )
            .unwrap();
            match path.forward_activity(&act).unwrap_err() {
                HybridError::InvalidConfig(msg) => {
                    assert!(msg.contains("outside tolerance"), "{label}: {msg}")
                }
                other => panic!("{label}: unexpected {other:?}"),
            }
            // Step 4 of `forward_activity` bumps the counter only after every
            // route invariant holds, so a rejected route leaves it at 0.
            assert_eq!(path.global_step(), 0, "{label}");
        }
    }

    /// An all-zero distribution gets its own diagnosis rather than falling
    /// through to the tolerance message.
    #[test]
    fn forward_activity_rejects_all_zero_weights() {
        let mut path = ReverseHybridPath::new(
            ProjectionMode::RateSum,
            4,
            8,
            FixedWeightsRouter::new(vec![0.0, 0.0], 1),
        )
        .unwrap();
        let act = SpikeActivity::from_fired(&[0], 4).unwrap();
        match path.forward_activity(&act).unwrap_err() {
            HybridError::InvalidConfig(msg) => assert!(msg.contains("all zero"), "{msg}"),
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(path.global_step(), 0);
    }

    /// The accept side: ordinary `f32` drift (three thirds sum to 1 + ~3e-8)
    /// must pass. This deviation is below what `tests/reverse_path.rs` already
    /// requires accepting, so it holds under any sane tolerance.
    #[test]
    fn forward_activity_accepts_normal_f32_drift() {
        let w = 1.0_f32 / 3.0;
        let mut path = ReverseHybridPath::new(
            ProjectionMode::RateSum,
            4,
            8,
            FixedWeightsRouter::new(vec![w, w, w], 2),
        )
        .unwrap();
        let act = SpikeActivity::from_fired(&[0], 4).unwrap();
        let out = path.forward_activity(&act).unwrap();

        let weights = out.expert_weights.expect("MoE weights");
        let sum: f64 = weights.iter().map(|&x| f64::from(x)).sum();
        assert!(
            (sum - 1.0).abs() <= WEIGHT_SUM_TOLERANCE,
            "renormalized sum {sum}"
        );
        assert_eq!(path.global_step(), 1);
    }

    /// Renormalization is observable: a sum inside the tolerance but several
    /// `f32` ULPs off 1.0 comes back rescaled, not passed through.
    #[test]
    fn forward_activity_renormalizes_weights_inside_tolerance() {
        // sum = 1.0 + ~5e-7: inside WEIGHT_SUM_TOLERANCE (9.54e-7) yet ~8 ULPs
        // of 0.5, so the rescale changes the emitted weights.
        let mut path = ReverseHybridPath::new(
            ProjectionMode::RateSum,
            4,
            8,
            FixedWeightsRouter::new(vec![0.5, 0.500_000_5], 2),
        )
        .unwrap();
        let act = SpikeActivity::from_fired(&[0], 4).unwrap();
        let out = path.forward_activity(&act).unwrap();

        let weights = out.expert_weights.expect("MoE weights");
        assert!(weights[0] < 0.5, "weights not renormalized: {weights:?}");
        let sum: f64 = weights.iter().map(|&x| f64::from(x)).sum();
        assert!(
            (sum - 1.0).abs() <= WEIGHT_SUM_TOLERANCE,
            "renormalized sum {sum}"
        );
        assert_eq!(path.global_step(), 1);
    }

    #[test]
    fn last_fired_prefers_last_non_empty_step() {
        let train = vec![vec![0], vec![], vec![1, 2]];
        assert_eq!(last_fired(&train), vec![1, 2]);
        let empty: Vec<Vec<usize>> = vec![vec![], vec![]];
        assert!(last_fired(&empty).is_empty());
    }

    #[test]
    fn accessors_match_construction() {
        let r = MockRouter::new(8, 3);
        let path = ReverseHybridPath::new(ProjectionMode::MembraneSnapshot, 16, 32, r).unwrap();
        assert_eq!(path.mode(), ProjectionMode::MembraneSnapshot);
        assert_eq!(path.n_neurons(), 16);
        assert_eq!(path.embed_dim(), 32);
        assert_eq!(path.router().num_experts(), 8);
        assert_eq!(path.router().top_k(), 3);
    }
}
