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

    pub fn mode(&self) -> ProjectionMode {
        self.mode
    }

    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }

    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    pub fn router(&self) -> &R {
        &self.router
    }

    pub fn router_mut(&mut self) -> &mut R {
        &mut self.router
    }

    pub fn global_step(&self) -> u64 {
        self.global_step
    }

    pub fn reset(&mut self) {
        self.global_step = 0;
    }

    /// Project activity, route through MoE, return `HybridOutput` with MoE fields set.
    ///
    /// Semantics match corinth-canal `Model::forward_activity` (projector + router half):
    /// 1. `global_step = saturating_add(1)`
    /// 2. `embedding = project_spike_activity(...)`
    /// 3. `route = router.route(&embedding)` — **no Sentry capture** on reverse v1
    /// 4. Build `HybridOutput` (empty `stimuli`; `fired_neurons` = last non-empty spike step)
    pub fn forward_activity(&mut self, activity: &SpikeActivity) -> Result<HybridOutput> {
        self.global_step = self.global_step.saturating_add(1);

        let embedding =
            project_spike_activity(self.mode, activity, self.n_neurons, self.embed_dim)?;

        // v1: propagate router errors without Sentry (validation-heavy path;
        // avoid flooding on bad activity / empty embeddings).
        let route = self.router.route(&embedding)?;

        let fired_neurons = last_fired(&activity.spike_train);

        Ok(HybridOutput {
            embedding,
            stimuli: Vec::new(),
            fired_neurons,
            global_step: self.global_step,
            expert_weights: Some(route.expert_weights),
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
                top_k: top_k.min(num_experts),
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
