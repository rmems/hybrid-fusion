// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration tests for reverse-path orchestration (`ReverseHybridPath`).
//!
//! File-free only: local mock `ExpertRouter` always; optional
//! `SyntheticExpertRouter` when feature `backends` is enabled.
//! No machine-local paths or network.

use hybrid_fusion::{
    ExpertRouteOutput, ExpertRouter, HybridError, ProjectionMode, Result, ReverseHybridPath,
    SpikeActivity,
};

// ---------------------------------------------------------------------------
// Mock ExpertRouter (always available)
// ---------------------------------------------------------------------------

struct MockRouter {
    num_experts: usize,
    top_k: usize,
}

impl MockRouter {
    fn new(num_experts: usize, top_k: usize) -> Self {
        assert!(num_experts > 0);
        Self {
            num_experts,
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
                "MockRouter: embedding must be non-empty".into(),
            ));
        }
        // Embedding-dependent but deterministic selection: pick experts by
        // absolute magnitude of successive embedding chunks.
        let n = self.num_experts;
        let mut scores = vec![0.0f32; n];
        for (i, &v) in embedding.iter().enumerate() {
            scores[i % n] += v.abs();
        }
        // Softmax-ish normalize for weight sum ≈ 1
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let expert_weights: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        let mut order: Vec<usize> = (0..n).collect();
        // total_cmp avoids Option/Ordering::Equal pitfalls with NaN.
        order.sort_by(|&a, &b| expert_weights[b].total_cmp(&expert_weights[a]));
        let selected_experts: Vec<usize> = order.into_iter().take(self.top_k).collect();

        Ok(ExpertRouteOutput {
            expert_weights,
            selected_experts,
            routing_entropy: Some(0.42),
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn assert_moe_ok(
    out: &hybrid_fusion::HybridOutput,
    embed_dim: usize,
    num_experts: usize,
    top_k: usize,
) {
    assert_eq!(out.embedding.len(), embed_dim);
    for v in &out.embedding {
        assert!(v.is_finite());
        assert!(*v >= -1.0 && *v <= 1.0, "tanh-bounded embedding, got {v}");
    }
    assert!(
        out.stimuli.is_empty(),
        "reverse path has no ANN→SNN stimuli"
    );
    let weights = out
        .expert_weights
        .as_ref()
        .expect("expert_weights must be Some on reverse path");
    assert_eq!(weights.len(), num_experts);
    assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    for w in weights {
        assert!(w.is_finite() && *w >= 0.0);
    }
    let selected = out
        .selected_experts
        .as_ref()
        .expect("selected_experts must be Some");
    assert_eq!(selected.len(), top_k);
    let mut seen = std::collections::HashSet::new();
    for &idx in selected {
        assert!(idx < num_experts);
        assert!(seen.insert(idx), "duplicate selected expert {idx}");
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn reverse_path_happy_rate_sum_mock_router() {
    let n_neurons = 8;
    let embed_dim = 16;
    let num_experts = 4;
    let top_k = 2;

    let router = MockRouter::new(num_experts, top_k);
    let mut path =
        ReverseHybridPath::new(ProjectionMode::RateSum, n_neurons, embed_dim, router).unwrap();

    let activity = SpikeActivity::from_fired(&[0, 2, 5], n_neurons).unwrap();
    let out = path.forward_activity(&activity).unwrap();

    assert_moe_ok(&out, embed_dim, num_experts, top_k);
    assert_eq!(out.fired_neurons, vec![0, 2, 5]);
    assert_eq!(out.global_step, 1);
    assert!(out.routing_entropy.is_some());
}

#[test]
fn reverse_path_membrane_snapshot_still_routes() {
    let n_neurons = 4;
    let embed_dim = 8;
    let router = MockRouter::new(3, 1);
    let mut path = ReverseHybridPath::new(
        ProjectionMode::MembraneSnapshot,
        n_neurons,
        embed_dim,
        router,
    )
    .unwrap();

    let activity = SpikeActivity {
        spike_train: vec![vec![1]],
        potentials: vec![0.1, 0.5, -0.2, 0.9],
        iz_potentials: vec![],
    };
    let out = path.forward_activity(&activity).unwrap();
    assert_moe_ok(&out, embed_dim, 3, 1);
    assert_eq!(out.fired_neurons, vec![1]);
}

#[test]
fn reverse_path_determinism_same_activity_twice() {
    let n_neurons = 6;
    let embed_dim = 12;
    let router = MockRouter::new(5, 2);
    let mut path =
        ReverseHybridPath::new(ProjectionMode::RateSum, n_neurons, embed_dim, router).unwrap();

    let activity = SpikeActivity::from_fired(&[0, 1, 4], n_neurons).unwrap();
    let a = path.forward_activity(&activity).unwrap();
    // New path for independent step counter; same router math
    let router2 = MockRouter::new(5, 2);
    let mut path2 =
        ReverseHybridPath::new(ProjectionMode::RateSum, n_neurons, embed_dim, router2).unwrap();
    let b = path2.forward_activity(&activity).unwrap();

    assert_eq!(a.embedding, b.embedding);
    assert_eq!(a.expert_weights, b.expert_weights);
    assert_eq!(a.selected_experts, b.selected_experts);
}

#[test]
fn reverse_path_rejects_bad_activity_from_projector() {
    let router = MockRouter::new(2, 1);
    let mut path = ReverseHybridPath::new(ProjectionMode::RateSum, 4, 8, router).unwrap();
    // potentials length mismatch vs n_neurons
    let bad = SpikeActivity {
        spike_train: vec![vec![0]],
        potentials: vec![0.0, 0.0], // len 2 != 4
        iz_potentials: vec![],
    };
    assert!(path.forward_activity(&bad).is_err());
    // global_step still increments before project in current design — verify
    // design: step increments first, then project. Spec §5.3 step 1 then 2.
    // If project fails after increment, step is 1.
    assert_eq!(path.global_step(), 1);
}

#[test]
fn reverse_path_new_rejects_zeros() {
    assert!(ReverseHybridPath::new(ProjectionMode::RateSum, 0, 8, MockRouter::new(2, 1)).is_err());
    assert!(ReverseHybridPath::new(ProjectionMode::RateSum, 4, 0, MockRouter::new(2, 1)).is_err());
}

#[cfg(feature = "backends")]
mod with_backends {
    use super::*;
    use hybrid_fusion::SyntheticExpertRouter;

    #[test]
    fn reverse_path_synthetic_router_moe_fields() {
        let n_neurons = 8;
        let embed_dim = 16;
        let num_experts = 4;
        let top_k = 2;
        let router = SyntheticExpertRouter::new(num_experts, top_k).unwrap();
        let mut path =
            ReverseHybridPath::new(ProjectionMode::RateSum, n_neurons, embed_dim, router).unwrap();
        let activity = SpikeActivity::from_fired(&[1, 3, 7], n_neurons).unwrap();
        let out = path.forward_activity(&activity).unwrap();
        assert_moe_ok(&out, embed_dim, num_experts, top_k);
        assert!(out.routing_entropy.is_some());
    }
}
