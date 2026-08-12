// SPDX-License-Identifier: MIT OR Apache-2.0

//! Synthetic [`ExpertRouter`](crate::ExpertRouter) using pure [`crate::routing`] math.
//!
//! No model files: gate scores from embedding chunks, softmax, top-k selection.

use crate::error::{HybridError, Result};
use crate::routing::{MAX_REASONABLE_EXPERTS, route_synthetic};
use crate::traits::{ExpertRouteOutput, ExpertRouter};

/// MoE router driven by [`route_synthetic`] (corinth pure DenseSim spirit without weights).
#[derive(Debug, Clone)]
pub struct SyntheticExpertRouter {
    num_experts: usize,
    top_k: usize,
}

impl SyntheticExpertRouter {
    /// `num_experts` and `top_k` must be ≥ 1; `top_k` is clamped to `num_experts`.
    pub fn new(num_experts: usize, top_k: usize) -> Result<Self> {
        if num_experts == 0 {
            return Err(HybridError::InvalidConfig(
                "SyntheticExpertRouter: num_experts must be >= 1".into(),
            ));
        }
        if num_experts > MAX_REASONABLE_EXPERTS {
            return Err(HybridError::InvalidConfig(format!(
                "SyntheticExpertRouter: num_experts ({num_experts}) exceeds max {MAX_REASONABLE_EXPERTS}"
            )));
        }
        if top_k == 0 {
            return Err(HybridError::InvalidConfig(
                "SyntheticExpertRouter: top_k must be >= 1".into(),
            ));
        }
        Ok(Self {
            num_experts,
            top_k: top_k.min(num_experts),
        })
    }
}

impl ExpertRouter for SyntheticExpertRouter {
    fn num_experts(&self) -> usize {
        self.num_experts
    }

    fn top_k(&self) -> usize {
        self.top_k
    }

    fn route(&mut self, embedding: &[f32]) -> Result<ExpertRouteOutput> {
        let (expert_weights, selected_experts, entropy) =
            route_synthetic(embedding, self.num_experts, self.top_k)?;
        Ok(ExpertRouteOutput {
            expert_weights,
            selected_experts,
            routing_entropy: Some(entropy),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_router_routes() {
        let mut r = SyntheticExpertRouter::new(4, 2).unwrap();
        let out = r.route(&[1.0, 0.5, 0.0, 2.0]).unwrap();
        assert_eq!(out.expert_weights.len(), 4);
        assert!((out.expert_weights.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert_eq!(out.selected_experts.len(), 2);
        assert!(out.routing_entropy.is_some());
    }
}
