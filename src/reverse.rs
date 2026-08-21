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
    /// 3. Validate `ExpertRouteOutput` invariants
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
        // f32 softmax/accumulation can drift for large expert counts, so
        // renormalize in f64 before populating HybridOutput. This keeps valid
        // trait-backed routers usable without exposing materially unnormalized
        // weights.
        let weights_sum: f64 = route.expert_weights.iter().map(|&w| w as f64).sum();
        if !weights_sum.is_finite() || weights_sum <= 0.0 {
            return Err(HybridError::InvalidConfig(
                "ReverseHybridPath: expert_weights sum is non-finite or non-positive".into(),
            ));
        }
        let scale = 1.0 / weights_sum;
        let expert_weights: Vec<f32> = route
            .expert_weights
            .iter()
            .map(|&w| (w as f64 * scale) as f32)
            .collect();
        let renorm_sum: f64 = expert_weights.iter().map(|&w| w as f64).sum();
        if (renorm_sum - 1.0).abs() > 1e-4 {
            return Err(HybridError::InvalidConfig(format!(
                "ReverseHybridPath: renormalized expert_weights sum {renorm_sum} is not within 1e-4 of 1.0"
            )));
        }
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
