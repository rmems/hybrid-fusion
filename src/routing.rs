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

/// Synthetic gate scores from an embedding and expert count.
///
/// Partitions the embedding across experts so **every** element is used:
/// base chunk size `width / num_experts`, with the first `width % num_experts`
/// experts getting one extra element. When `num_experts > width`, the first
/// `width` experts each get one element and the rest score `0.0`.
///
/// Empty embedding → zeros for each expert. Zero experts → empty vec.
pub fn synthetic_gate_scores(num_experts: usize, embedding: &[f32]) -> Vec<f32> {
    if num_experts == 0 {
        return Vec::new();
    }
    if embedding.is_empty() {
        return vec![0.0; num_experts];
    }
    let width = embedding.len();
    let base = width / num_experts;
    let rem = width % num_experts;
    let mut gate_scores = Vec::with_capacity(num_experts);
    let mut start = 0;
    for expert_id in 0..num_experts {
        let len = base + usize::from(expert_id < rem);
        if len == 0 {
            gate_scores.push(0.0);
        } else {
            let end = start + len;
            gate_scores.push(embedding[start..end].iter().sum());
            start = end;
        }
    }
    debug_assert_eq!(start, width);
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

/// Indices of the `top_k` largest weights (descending). NaN weights sort last.
///
/// Returns at most `weights.len()` indices. `top_k == 0` yields empty.
pub fn top_k_indices(weights: &[f32], top_k: usize) -> Vec<usize> {
    if top_k == 0 || weights.is_empty() {
        return Vec::new();
    }
    let mut indexed: Vec<(usize, f32)> = weights.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| match (a.1.is_nan(), b.1.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater, // NaN after finite
        (false, true) => Ordering::Less,
        (false, false) => b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal),
    });
    indexed
        .into_iter()
        .take(top_k.min(weights.len()))
        .map(|(idx, _)| idx)
        .collect()
}

/// Shannon entropy of a discrete distribution, normalized by `ln(n)` into `[0, 1]`.
///
/// Accumulates in `f64` for large expert counts. Empty or single-weight → `0.0`.
pub fn routing_entropy(weights: &[f32]) -> f32 {
    if weights.len() <= 1 {
        return 0.0;
    }
    let entropy: f64 = weights
        .iter()
        .copied()
        .filter(|w| w.is_finite() && *w > 0.0)
        .map(|w| {
            let w = f64::from(w);
            -w * w.ln()
        })
        .sum();
    let max_entropy = (weights.len() as f64).ln();
    if max_entropy > 0.0 {
        (entropy / max_entropy).clamp(0.0, 1.0) as f32
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
/// - `num_experts > MAX_REASONABLE_EXPERTS`
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
    if num_experts > MAX_REASONABLE_EXPERTS {
        return Err(HybridError::InvalidConfig(format!(
            "route_synthetic: num_experts ({num_experts}) exceeds max {MAX_REASONABLE_EXPERTS}"
        )));
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
    fn top_k_nan_sorts_last() {
        let idx = top_k_indices(&[0.1, f32::NAN, 0.9], 2);
        assert_eq!(idx, vec![2, 0]);
        assert!(!idx.contains(&1));
    }

    #[test]
    fn synthetic_scores_uses_full_embedding() {
        // len 5, 2 experts → chunks [0..3] and [3..5] (remainder to first)
        let emb = [1.0, 1.0, 1.0, 10.0, 10.0];
        let s = synthetic_gate_scores(2, &emb);
        assert_eq!(s.len(), 2);
        assert!((s[0] - 3.0).abs() < 1e-5);
        assert!((s[1] - 20.0).abs() < 1e-5);
    }

    #[test]
    fn synthetic_scores_more_experts_than_dims() {
        let emb = [1.0, 2.0];
        let s = synthetic_gate_scores(4, &emb);
        assert_eq!(s, vec![1.0, 2.0, 0.0, 0.0]);
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
    fn route_synthetic_rejects_empty_and_huge() {
        assert!(route_synthetic(&[], 2, 1).is_err());
        assert!(route_synthetic(&[1.0], 0, 1).is_err());
        assert!(route_synthetic(&[1.0], MAX_REASONABLE_EXPERTS + 1, 1).is_err());
    }
}
