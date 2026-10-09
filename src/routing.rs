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
/// # Non-finite policy
///
/// Each case takes the limit of softmax for the scores it is given. No case
/// falls back to a uniform distribution to hide a bad score.
///
/// - **Any `NaN`** → `Err(InvalidConfig)` naming the first `NaN` index, even if
///   other scores are `±Inf`. A `NaN` score has no defined weight.
/// - **One or more `+Inf`** → each `+Inf` index gets `1 / k` of the mass
///   (`k` = number of `+Inf` scores) and every other index gets exactly `0.0`.
///   A single `+Inf` gives a one-hot result.
/// - **All `-Inf`** → uniform `1 / n`. All scores are equal, so this is the
///   limit of `softmax([c; n])` as `c → -∞`. Its normalized entropy is `1.0`,
///   so callers that need to detect "no expert scored" should check the scores.
/// - **Some `-Inf` with a finite max** → `-Inf` indices get exactly `0.0`.
/// - Empty input → `Ok(vec![])`.
///
/// Large finite scores (e.g. `1e30`) take the normal max-subtracted path.
///
/// # Errors
///
/// [`HybridError::InvalidConfig`] if any score is `NaN`.
pub fn softmax(scores: &[f32]) -> Result<Vec<f32>> {
    if let Some(index) = scores.iter().position(|s| s.is_nan()) {
        return Err(HybridError::InvalidConfig(format!(
            "softmax: score[{index}] is NaN (of {} scores); NaN scores are rejected, not \
             sanitized",
            scores.len()
        )));
    }
    if scores.is_empty() {
        return Ok(Vec::new());
    }
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_score == f32::INFINITY {
        let k = scores.iter().filter(|&&s| s == f32::INFINITY).count();
        let share = (1.0 / k as f64) as f32;
        return Ok(scores
            .iter()
            .map(|&s| if s == f32::INFINITY { share } else { 0.0 })
            .collect());
    }
    if max_score == f32::NEG_INFINITY {
        // NaN is excluded above, so every score is -Inf.
        let u = (1.0 / scores.len() as f64) as f32;
        return Ok(vec![u; scores.len()]);
    }
    let exp_scores: Vec<f32> = scores
        .iter()
        .map(|&score| (score - max_score).exp())
        .collect();
    // The normalizer MUST be accumulated in f64. Naive f32 summation of `n`
    // non-negative terms carries relative error up to `n * u` (u = 2^-24), and
    // that error lands on the denominator, so it passes essentially undivided
    // into the sum of the returned weights: at MAX_REASONABLE_EXPERTS the
    // weights re-accumulate to 1 +/- ~1.5e-2 (worse for score shapes whose tail
    // absorbs into the running sum, measured up to 4.8e-2). No downstream
    // tolerance can tell that apart from a genuinely unnormalized distribution.
    // In f64 the denominator is exact to ~n * 2^-53, leaving one f32 rounding
    // per weight -- a bound independent of `n`.
    //
    // With a finite max, the max term is exp(0) = 1 and every term lies in
    // [0, 1], so 1 <= sum_exp <= n. No fallback is needed.
    let sum_exp: f64 = exp_scores.iter().map(|&v| f64::from(v)).sum();
    debug_assert!(sum_exp >= 1.0 && sum_exp.is_finite());
    Ok(exp_scores
        .into_iter()
        .map(|v| (f64::from(v) / sum_exp) as f32)
        .collect())
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
/// - Any gate score is `NaN`. This happens when the embedding contains `NaN`,
///   or when `+Inf` and `-Inf` fall in the same expert's chunk.
///
/// Gate scores follow the [`softmax`] non-finite policy. An expert whose chunk
/// sums to `+Inf` (an `Inf` element, or finite overflow) takes all the weight,
/// split equally with any other `+Inf` experts.
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
    if let Some(expert) = scores.iter().position(|s| s.is_nan()) {
        return Err(HybridError::InvalidConfig(format!(
            "route_synthetic: gate score for expert {expert} is NaN (embedding contains NaN, \
             or +Inf and -Inf in the same chunk)"
        )));
    }
    let weights = softmax(&scores)?;
    let selected = top_k_indices(&weights, top_k.min(num_experts));
    let entropy = routing_entropy(&weights);
    Ok((weights, selected, entropy))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let w = softmax(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(w.len(), 3);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(w[2] > w[1] && w[1] > w[0]);
    }

    /// Locks the invariant [`crate::ReverseHybridPath`] relies on: re-accumulating
    /// the returned `f32` weights in `f64` stays within a couple of `f32`
    /// roundings of `1.0` at **every** expert count and score shape.
    ///
    /// The adversarial shape is two-level -- a small head at the max and a long
    /// tail just small enough that its terms absorb into a running `f32` sum.
    /// With an `f32` denominator this drifts to ~4.8e-2 at
    /// [`MAX_REASONABLE_EXPERTS`]; with the `f64` denominator it stays under `2u`.
    #[test]
    fn softmax_sum_stays_within_f32_roundoff_at_every_scale() {
        const U: f64 = 5.960_464_477_539_063e-8; // f32::EPSILON / 2
        for n in [1_000usize, 100_000, MAX_REASONABLE_EXPERTS] {
            let hi = (n / 50).max(1); // 2% head: the measured worst case
            let m = -((hi as f64 * U).ln()) as f32;
            let mut scores = vec![-m; n];
            for s in scores.iter_mut().take(hi) {
                *s = 0.0;
            }
            let sum: f64 = softmax(&scores)
                .unwrap()
                .iter()
                .map(|&w| f64::from(w))
                .sum();
            assert!(
                (sum - 1.0).abs() <= 2.0 * U,
                "n={n}: f64-re-accumulated softmax sum {sum} drifts {:e} (> 2u)",
                (sum - 1.0).abs()
            );
        }
    }

    #[test]
    fn softmax_empty_is_empty() {
        assert!(softmax(&[]).unwrap().is_empty());
    }

    #[test]
    fn softmax_single_pos_inf_is_one_hot_on_that_index() {
        assert_eq!(
            softmax(&[f32::INFINITY, 0.0, 0.0]).unwrap(),
            vec![1.0, 0.0, 0.0]
        );
        // Not at index 0, alongside a larger finite score and a -Inf.
        assert_eq!(
            softmax(&[1e30, -3.0, f32::INFINITY, f32::NEG_INFINITY]).unwrap(),
            vec![0.0, 0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn softmax_multiple_pos_inf_split_mass_only_among_them() {
        let w = softmax(&[f32::INFINITY, 5.0, f32::INFINITY, 0.0, f32::INFINITY]).unwrap();
        let third = (1.0f64 / 3.0) as f32;
        assert_eq!(w, vec![third, 0.0, third, 0.0, third]);
        let sum: f64 = w.iter().map(|&v| f64::from(v)).sum();
        assert!((sum - 1.0).abs() <= crate::WEIGHT_SUM_TOLERANCE);
    }

    #[test]
    fn softmax_rejects_any_nan() {
        for scores in [
            vec![f32::NAN, 1.0],
            vec![1.0, 2.0, f32::NAN],
            vec![f32::NAN, f32::NAN],
            // NaN must not be masked by the +Inf or all--Inf branches.
            vec![f32::INFINITY, f32::NAN],
            vec![f32::NEG_INFINITY, f32::NAN],
        ] {
            assert!(
                matches!(softmax(&scores), Err(HybridError::InvalidConfig(_))),
                "{scores:?} must be rejected"
            );
        }
    }

    #[test]
    fn softmax_all_neg_inf_is_uniform() {
        assert_eq!(softmax(&[f32::NEG_INFINITY; 4]).unwrap(), vec![0.25; 4]);
    }

    #[test]
    fn softmax_partial_neg_inf_gets_exact_zero() {
        let w = softmax(&[f32::NEG_INFINITY, 0.0, 0.0]).unwrap();
        assert_eq!(w, vec![0.0, 0.5, 0.5]);
    }

    #[test]
    fn softmax_large_finite_is_one_hot_and_normalized() {
        let w = softmax(&[0.0, 1e30, -1e30]).unwrap();
        assert_eq!(w, vec![0.0, 1.0, 0.0]);
        let sum: f64 = w.iter().map(|&v| f64::from(v)).sum();
        assert!((sum - 1.0).abs() <= crate::WEIGHT_SUM_TOLERANCE);
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

    #[test]
    fn route_synthetic_rejects_nan_gate_scores() {
        // NaN element in expert 1's chunk.
        assert!(matches!(
            route_synthetic(&[1.0, 0.0, f32::NAN, 0.0], 2, 1),
            Err(HybridError::InvalidConfig(_))
        ));
        // +Inf and -Inf in the same chunk sum to NaN.
        assert!(matches!(
            route_synthetic(&[f32::INFINITY, f32::NEG_INFINITY, 1.0, 2.0], 2, 1),
            Err(HybridError::InvalidConfig(_))
        ));
    }

    #[test]
    fn route_synthetic_pos_inf_chunk_takes_all_weight() {
        // Expert 2 sees +Inf; expert 0 has the largest finite score.
        let (w, sel, h) = route_synthetic(&[9.0, 0.0, f32::INFINITY, 0.0], 4, 2).unwrap();
        assert_eq!(w, vec![0.0, 0.0, 1.0, 0.0]);
        assert_eq!(sel[0], 2);
        assert_eq!(h, 0.0);
    }
}
