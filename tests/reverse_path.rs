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

/// Router that always fails, so router errors can be observed on a valid
/// (non-empty) embedding — `MockRouter` only rejects empty embeddings, which
/// the projector never produces for `embed_dim > 0`.
struct FailingRouter;

impl ExpertRouter for FailingRouter {
    fn num_experts(&self) -> usize {
        2
    }

    fn top_k(&self) -> usize {
        1
    }

    fn route(&mut self, _embedding: &[f32]) -> Result<ExpertRouteOutput> {
        Err(HybridError::InvalidConfig(
            "FailingRouter: always fails".into(),
        ))
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
    // global_step now only increments after projection and routing succeed,
    // so a failed projector call leaves the counter at 0.
    assert_eq!(path.global_step(), 0);
}

/// RM-1445: a timestep is a set — a step naming the same neuron twice is
/// rejected instead of being counted twice into a rate above 1.0 — and a
/// non-empty `iz_potentials` whose length is not `n_neurons` is a shape
/// error rather than a silently ignored field.
#[test]
fn reverse_path_rejects_duplicate_spikes_and_garbage_iz_potentials() {
    let router = MockRouter::new(2, 1);
    let mut path = ReverseHybridPath::new(ProjectionMode::RateSum, 4, 8, router).unwrap();

    // Duplicate fired IDs in one step (the from_fired multiset case).
    let duplicated = SpikeActivity {
        spike_train: vec![vec![0, 0, 0]],
        potentials: vec![0.0; 4],
        iz_potentials: vec![],
    };
    let err = path.forward_activity(&duplicated).unwrap_err();
    match err {
        HybridError::InvalidConfig(msg) => {
            assert!(
                msg.contains("index 0 appears more than once at step 0"),
                "{msg}"
            );
        }
        other => panic!("expected duplicate rejection, got {other:?}"),
    }
    assert_eq!(path.global_step(), 0);

    // Well-formed (neuron-sized) iz_potentials is accepted and ignored.
    let ignored = SpikeActivity {
        spike_train: vec![vec![0]],
        potentials: vec![0.0; 4],
        iz_potentials: vec![0.5; 4],
    };
    assert!(path.forward_activity(&ignored).is_ok());
    assert_eq!(path.global_step(), 1);

    // Garbage-length iz_potentials is a shape mismatch, not a silent ignore.
    let garbage = SpikeActivity {
        spike_train: vec![vec![0]],
        potentials: vec![0.0; 4],
        iz_potentials: vec![0.5; 99],
    };
    match path.forward_activity(&garbage).unwrap_err() {
        HybridError::InputLengthMismatch {
            expected: 4,
            got: 99,
        } => {}
        other => panic!("expected iz_potentials length mismatch, got {other:?}"),
    }
    assert_eq!(
        path.global_step(),
        1,
        "the rejected activity is not counted"
    );
}

#[test]
fn reverse_path_propagates_router_errors() {
    let mut path = ReverseHybridPath::new(ProjectionMode::RateSum, 4, 8, FailingRouter).unwrap();
    let activity = SpikeActivity::from_fired(&[0, 3], 4).unwrap();

    // Projection succeeds (embed_dim > 0), so this is the router's error.
    let err = path.forward_activity(&activity).unwrap_err();
    match err {
        HybridError::InvalidConfig(msg) => assert!(msg.contains("FailingRouter")),
        other => panic!("expected router InvalidConfig, got {other:?}"),
    }
    // global_step now only increments after projection and routing succeed,
    // so a failed router call leaves the counter at 0.
    assert_eq!(path.global_step(), 0);
}

#[test]
fn reverse_path_new_rejects_zeros() {
    assert!(ReverseHybridPath::new(ProjectionMode::RateSum, 0, 8, MockRouter::new(2, 1)).is_err());
    assert!(ReverseHybridPath::new(ProjectionMode::RateSum, 4, 0, MockRouter::new(2, 1)).is_err());
}

/// RM-1465: a finite reverse-path `HybridOutput` round-trips through JSON
/// without losing any MoE field, and a non-finite value fails closed with an
/// error naming the field instead of silently becoming `null`.
#[test]
fn reverse_path_output_json_policy_round_trips_and_fails_closed() {
    let n_neurons = 8;
    let embed_dim = 16;
    let mut path = ReverseHybridPath::new(
        ProjectionMode::RateSum,
        n_neurons,
        embed_dim,
        MockRouter::new(4, 2),
    )
    .unwrap();
    let activity = SpikeActivity::from_fired(&[1, 3, 7], n_neurons).unwrap();
    let out = path.forward_activity(&activity).unwrap();
    assert_moe_ok(&out, embed_dim, 4, 2);

    // Finite record: no `null` anywhere, every field survives.
    let json = serde_json::to_string(&out).expect("finite reverse-path output serializes");
    assert!(!json.contains("null"), "unexpected null in {json}");
    let restored: hybrid_fusion::HybridOutput =
        serde_json::from_str(&json).expect("finite reverse-path output round-trips");
    assert_eq!(restored.embedding, out.embedding);
    assert_eq!(restored.stimuli, out.stimuli);
    assert_eq!(restored.fired_neurons, out.fired_neurons);
    assert_eq!(restored.num_neurons, out.num_neurons);
    assert_eq!(restored.global_step, out.global_step);
    assert_eq!(restored.expert_weights, out.expert_weights);
    assert_eq!(restored.selected_experts, out.selected_experts);
    assert_eq!(restored.routing_entropy, out.routing_entropy);

    // Hand-built corruption (a router could still report non-finite entropy,
    // or a caller could build the record by hand): serialization fails closed.
    let mut bad = out;
    bad.routing_entropy = Some(f32::NAN);
    let err = serde_json::to_string(&bad)
        .expect_err("non-finite entropy must fail closed, not write null");
    let text = err.to_string();
    assert!(
        text.starts_with("HybridOutput.routing_entropy is NaN"),
        "unexpected error: {text}"
    );
    assert!(
        text.contains("JSON would emit null"),
        "policy reason in {text}"
    );
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
