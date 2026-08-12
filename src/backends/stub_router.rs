// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stub [`ExpertRouter`](crate::ExpertRouter) for tests and examples.
//!
//! Produces a uniform gate distribution and selects the first `top_k` experts.
//! No checkpoints, no matmul — pure bookkeeping for reverse-path smoke tests.

use crate::error::{HybridError, Result};
use crate::traits::{ExpertRouteOutput, ExpertRouter};

/// Uniform / stub MoE router (corinth-canal `RoutingMode::StubUniform` spirit).
#[derive(Debug, Clone)]
pub struct StubExpertRouter {
    num_experts: usize,
    top_k: usize,
}

impl StubExpertRouter {
    /// Build a stub router. `num_experts` and `top_k` must be ≥ 1; `top_k` is
    /// clamped to `num_experts`.
    pub fn new(num_experts: usize, top_k: usize) -> Result<Self> {
        if num_experts == 0 {
            return Err(HybridError::InvalidConfig(
                "StubExpertRouter: num_experts must be >= 1".into(),
            ));
        }
        if top_k == 0 {
            return Err(HybridError::InvalidConfig(
                "StubExpertRouter: top_k must be >= 1".into(),
            ));
        }
        Ok(Self {
            num_experts,
            top_k: top_k.min(num_experts),
        })
    }
}

impl ExpertRouter for StubExpertRouter {
    fn num_experts(&self) -> usize {
        self.num_experts
    }

    fn top_k(&self) -> usize {
        self.top_k
    }

    fn route(&mut self, embedding: &[f32]) -> Result<ExpertRouteOutput> {
        if embedding.is_empty() {
            return Err(HybridError::InputLengthMismatch {
                expected: 1,
                got: 0,
            });
        }
        let w = 1.0 / self.num_experts as f32;
        let expert_weights = vec![w; self.num_experts];
        let selected_experts: Vec<usize> = (0..self.top_k).collect();
        // Uniform discrete distribution: H = log(n) nats, normalized later by callers if needed.
        let routing_entropy = Some((self.num_experts as f32).ln());
        Ok(ExpertRouteOutput {
            expert_weights,
            selected_experts,
            routing_entropy,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_route_uniform_and_top_k() {
        let mut router = StubExpertRouter::new(4, 2).unwrap();
        assert_eq!(router.num_experts(), 4);
        assert_eq!(router.top_k(), 2);

        let out = router.route(&[0.1, 0.2, 0.3]).unwrap();
        assert_eq!(out.expert_weights.len(), 4);
        assert!((out.expert_weights.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert_eq!(out.selected_experts, vec![0, 1]);
        assert!(out.routing_entropy.is_some());
    }

    #[test]
    fn stub_rejects_empty_embedding() {
        let mut router = StubExpertRouter::new(2, 1).unwrap();
        let err = router.route(&[]).unwrap_err();
        match err {
            HybridError::InputLengthMismatch { got: 0, .. } => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn stub_clamps_top_k() {
        let router = StubExpertRouter::new(3, 10).unwrap();
        assert_eq!(router.top_k(), 3);
    }
}
