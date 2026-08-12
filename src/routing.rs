// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pure MoE routing math for [`ExpertRouter`](crate::ExpertRouter) backends.
//!
//! Extracted from corinth-canal `src/moe/routing.rs` patterns (`synthetic_gate_scores`,
//! softmax, top-k). **No** checkpoint matmul / GGUF / Safetensors I/O.
//!
//! Use these helpers to build deterministic, file-free routers (see
//! [`crate::backends::stub_router::StubExpertRouter`] and
//! [`crate::backends::synthetic_router::SyntheticExpertRouter`]).

use std::cmp::Ordering;

use crate::error::{HybridError, Result};

/// Soft upper bound for synthetic / stub expert counts (keeps f32 math sane).
pub const MAX_REASONABLE_EXPERTS: usize = 1_000_000;

/// Synthetic gate scores from an embedding and expert count (corinth-canal).
///
/// Partitions the embedding into chunks (with wrap) and sums each chunk as the
/// score for one expert. Empty embedding or zero experts yields an empty vec.
pub fn synthetic_gate_scores(num_experts: usize, embedding: &[f32]) -> Vec<f32> {
    if num_experts == 0 {
        return Vec::new();
    }
    let width = embedding.len().max(1);
    let chunk = (width / num_experts).max(1);
    let mut gate_scores = Vec::with_capacity(num_experts);
    if embedding.is_empty() {
        return vec![0.0; num_experts];
    }
    for expert_id in 0..num_experts {
        let start = (expert_id * chunk) % width;
        let end = (start + chunk).min(width);
        gate_scores.push(embedding[start..end].iter().sum());
    }
    gate_scores
}

/// Numerically stable softmax over scores (sums to ~1 when non-empty).
///
/// Non-finite or non-positive sum falls back to a uniform distribution.
pub fn softmax(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_score.is_finite() {
        let u = 1.0 / scores.len() as f32;
        return vec![u; scores.len()];
    }
    let exp_scores: Vec<f32> = scores
        .iter()
        .map(|&score| (score - max_score).exp())
        .collect();
    let sum_exp: f32 = exp_scores.iter().sum();
    if sum_exp <= 0.0 || !sum_exp.is_finite() {
        let u = 1.0 / scores.len() as f32;
        return vec![u; scores.len()];
    }
    exp_scores.into_iter().map(|v| v / sum_exp).collect()
}

/// Indices of the `top_k` largest weights (stable on ties via index order after sort).
///
/// Returns at most `weights.len()` indices. `top_k == 0` yields empty.
pub fn top_k_indices(weights: &[f32], top_k: usize) -> Vec<usize> {
    if top_k == 0 || weights.is_empty() {
        return Vec::new();
    }
    let mut indexed: Vec<(usize, f32)> = weights.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    indexed
        .into_iter()
        .take(top_k.min(weights.len()))
        .map(|(idx, _)| idx)
        .collect()
}

/// Shannon entropy of a discrete distribution, normalized by `ln(n)` into `[0, 1]`.
///
/// From corinth-canal `latent::normalized_entropy`. Empty or single-weight → `0.0`.
pub fn routing_entropy(weights: &[f32]) -> f32 {
    if weights.len() <= 1 {
        return 0.0;
    }
    let entropy = weights
        .iter()
        .copied()
        .filter(|w| w.is_finite() && *w > 0.0)
        .map(|w| -w * w.ln())
        .sum::<f32>();
    let max_entropy = (weights.len() as f32).ln();
    if max_entropy > 0.0 {
        (entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Full pure route: synthetic gates → softmax → top-k + entropy.
///
/// # Errors
///
/// - Empty embedding
/// - `num_experts == 0` or `top_k == 0`
pub fn route_synthetic(
    embedding: &[f32],
    num_experts: usize,
    top_k: usize,
) -> Result<(Vec<f32>, Vec<usize>, f32)> {
    if embedding.is_empty() {
        return Err(HybridError::InvalidConfig(
            "route_synthetic: embedding must be non-empty".into(),
        ));
    }
    if num_experts == 0 {
        return Err(HybridError::InvalidConfig(
            "route_synthetic: num_experts must be >= 1".into(),
        ));
    }
    if top_k == 0 {
        return Err(HybridError::InvalidConfig(
            "route_synthetic: top_k must be >= 1".into(),
        ));
    }
    let scores = synthetic_gate_scores(num_experts, embedding);
    let weights = softmax(&scores);
    let selected = top_k_indices(&weights, top_k.min(num_experts));
    let entropy = routing_entropy(&weights);
    Ok((weights, selected, entropy))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let w = softmax(&[1.0, 2.0, 3.0]);
        assert_eq!(w.len(), 3);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(w[2] > w[1] && w[1] > w[0]);
    }

    #[test]
    fn softmax_empty_and_uniform_fallback() {
        assert!(softmax(&[]).is_empty());
        let w = softmax(&[f32::NAN, f32::NAN]);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn top_k_deterministic() {
        let idx = top_k_indices(&[0.1, 0.5, 0.2, 0.2], 2);
        assert_eq!(idx, vec![1, 2]); // 0.5 first, then first 0.2 at index 2
        assert!(top_k_indices(&[1.0], 0).is_empty());
    }

    #[test]
    fn synthetic_scores_deterministic() {
        let emb = [1.0, 2.0, 3.0, 4.0];
        let a = synthetic_gate_scores(2, &emb);
        let b = synthetic_gate_scores(2, &emb);
        assert_eq!(a, b);
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn entropy_uniform_near_one() {
        let u = vec![0.25; 4];
        assert!((routing_entropy(&u) - 1.0).abs() < 1e-5);
        assert_eq!(routing_entropy(&[1.0]), 0.0);
    }

    #[test]
    fn route_synthetic_happy_path() {
        let (w, sel, h) = route_synthetic(&[1.0, 0.0, 2.0, 0.0], 4, 2).unwrap();
        assert_eq!(w.len(), 4);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert_eq!(sel.len(), 2);
        assert!((0.0..=1.0).contains(&h));
    }

    #[test]
    fn route_synthetic_rejects_empty() {
        assert!(route_synthetic(&[], 2, 1).is_err());
        assert!(route_synthetic(&[1.0], 0, 1).is_err());
    }
}
