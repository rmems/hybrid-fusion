// SPDX-License-Identifier: MIT OR Apache-2.0

//! Projector helpers.
//!
//! - **Forward (ANN→SNN):** [`embed_to_stimuli`] / [`embed_to_stimuli_with_width`]
//! - **Reverse (SNN→embedding→MoE):** [`project_spike_activity`] / [`ProjectionMode`]

use crate::error::{HybridError, Result};
use crate::tensor::Tensor;
use crate::traits::SpikeActivity;
use crate::types::ProjectionMode;

/// Temporal histogram bins (matches corinth-canal `TEMPORAL_BINS`).
const TEMPORAL_BINS: usize = 4;
/// Adaptive-bank dims when `iz_potentials` is short (corinth `IZ_NEURONS` spirit).
const IZ_FEATURE_DIM: usize = 5;

fn mean_pool(embedding: &Tensor) -> Vec<f32> {
    match embedding.ndim() {
        0 | 1 => embedding.data().to_vec(),
        2 => {
            let shape = embedding.shape();
            let seq = shape[0].max(1);
            let dim = shape[1];
            let data = embedding.data();
            let mut pooled = vec![0.0f32; dim];
            for t in 0..seq {
                let row = &data[t * dim..(t + 1) * dim];
                for (i, v) in row.iter().enumerate() {
                    pooled[i] += *v;
                }
            }
            let inv = 1.0 / seq as f32;
            for v in &mut pooled {
                *v *= inv;
            }
            pooled
        }
        _ => {
            // Treat higher-rank tensors as flat — no silent semantic change.
            embedding.data().to_vec()
        }
    }
}

pub(crate) fn resize_to(src: &[f32], target_width: usize) -> Vec<f32> {
    if target_width == 0 {
        return Vec::new();
    }
    let src_len = src.len();
    if src_len == 0 {
        return vec![0.0; target_width];
    }
    if src_len == target_width {
        return src.to_vec();
    }

    if src_len > target_width {
        let mut out = Vec::with_capacity(target_width);
        for i in 0..target_width {
            let start = (i * src_len) / target_width;
            let end = ((i + 1) * src_len) / target_width;
            let end = end.max(start + 1).min(src_len);
            let slice = &src[start..end];
            let mean = slice.iter().sum::<f32>() / slice.len() as f32;
            out.push(mean);
        }
        out
    } else {
        let mut out = Vec::with_capacity(target_width);
        out.extend_from_slice(src);
        out.resize(target_width, 0.0);
        out
    }
}

fn squash_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = x.tanh();
    }
}

pub fn embed_to_stimuli(embedding: &Tensor) -> Vec<f32> {
    let mut pooled = mean_pool(embedding);
    squash_inplace(&mut pooled);
    pooled
}

pub fn embed_to_stimuli_with_width(embedding: &Tensor, snn_width: usize) -> Vec<f32> {
    let pooled = mean_pool(embedding);
    let mut resized = resize_to(&pooled, snn_width);
    squash_inplace(&mut resized);
    resized
}

// ── Reverse path: SpikeActivity → dense embedding (pure modes) ─────────────

/// Build a mode-specific feature vector from spike activity (no learned weights).
///
/// Feature layout follows corinth-canal `build_feature_vector` blends, without
/// EMA state or Xavier W/b. Length is mode-dependent (≈ `n_neurons * (1+4+1) + 5`).
pub fn spike_activity_features(
    mode: ProjectionMode,
    activity: &SpikeActivity,
    n_neurons: usize,
) -> Result<Vec<f32>> {
    if n_neurons == 0 {
        return Err(HybridError::InvalidConfig(
            "spike_activity_features: n_neurons must be > 0".into(),
        ));
    }
    if activity.potentials.len() < n_neurons {
        return Err(HybridError::InputLengthMismatch {
            expected: n_neurons,
            got: activity.potentials.len(),
        });
    }
    for (t, step) in activity.spike_train.iter().enumerate() {
        if let Some(&idx) = step.iter().find(|&&i| i >= n_neurons) {
            return Err(HybridError::InvalidConfig(format!(
                "spike_activity_features: spike index {idx} out of range at step {t} (n_neurons={n_neurons})"
            )));
        }
    }

    let n_steps = activity.spike_train.len().max(1) as f32;

    let mut rates = vec![0.0_f32; n_neurons];
    for step in &activity.spike_train {
        for &idx in step {
            rates[idx] += 1.0;
        }
    }
    for r in &mut rates {
        *r /= n_steps;
    }

    let bins = TEMPORAL_BINS;
    let mut hist = vec![0.0_f32; n_neurons * bins];
    if !activity.spike_train.is_empty() {
        let steps = activity.spike_train.len();
        for (t, step) in activity.spike_train.iter().enumerate() {
            let bin = ((t * bins) / steps).min(bins - 1);
            for &idx in step {
                hist[idx * bins + bin] += 1.0;
            }
        }
        let total = (n_steps / bins as f32).max(1.0);
        for h in &mut hist {
            *h /= total;
        }
    }

    let membrane: Vec<f32> = activity.potentials[..n_neurons]
        .iter()
        .map(|&v| v.clamp(0.0, 1.0))
        .collect();

    let iz: Vec<f32> = activity
        .iz_potentials
        .iter()
        .take(IZ_FEATURE_DIM)
        .map(|&v| (v / 30.0).clamp(-1.0, 1.0))
        .chain(std::iter::repeat(0.0))
        .take(IZ_FEATURE_DIM)
        .collect();

    let mut features = Vec::new();
    match mode {
        ProjectionMode::RateSum | ProjectionMode::SpikingTernary => {
            // SpikingTernary pure path matches RateSum features; GIF lives elsewhere.
            features.extend_from_slice(&rates);
            features.extend_from_slice(&hist);
            features.extend_from_slice(&membrane);
            features.extend_from_slice(&iz);
        }
        ProjectionMode::TemporalHistogram => {
            features.extend(rates.iter().map(|r| r * 0.3));
            features.extend(hist.iter().map(|h| h * 2.0));
            features.extend_from_slice(&membrane);
            features.extend_from_slice(&iz);
        }
        ProjectionMode::MembraneSnapshot => {
            features.extend_from_slice(&rates);
            features.extend_from_slice(&hist);
            features.extend(membrane.iter().map(|v| v * 2.0));
            features.extend_from_slice(&iz);
        }
    }
    Ok(features)
}

/// Project spike activity into a dense embedding for **MoE ExpertRouter** input.
///
/// Pure: features → resize to `embed_dim` → `tanh` (no learned linear layer).
/// Does **not** implement SAAQ / latent calibration.
pub fn project_spike_activity(
    mode: ProjectionMode,
    activity: &SpikeActivity,
    n_neurons: usize,
    embed_dim: usize,
) -> Result<Vec<f32>> {
    if embed_dim == 0 {
        return Err(HybridError::InvalidConfig(
            "project_spike_activity: embed_dim must be > 0".into(),
        ));
    }
    let features = spike_activity_features(mode, activity, n_neurons)?;
    let mut embedding = resize_to(&features, embed_dim);
    squash_inplace(&mut embedding);
    Ok(embedding)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_to_stimuli_1d_bounded() {
        let t = Tensor::from_vec(vec![100.0, -100.0, 0.0, 50.0], &[4]);
        let out = embed_to_stimuli(&t);
        assert_eq!(out.len(), 4);
        for v in &out {
            assert!(v.abs() <= 1.0, "tanh must bound values in [-1, 1], got {v}");
        }
        assert!(out[0] > 0.99);
        assert!(out[1] < -0.99);
        assert!((out[2]).abs() < 1e-6);
    }

    #[test]
    fn test_embed_to_stimuli_2d_mean_pool() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0], &[2, 3]);
        let out = embed_to_stimuli(&t);
        assert_eq!(out.len(), 3);
        assert!((out[0] - 2.0f32.tanh()).abs() < 1e-5);
        assert!((out[1] - 3.0f32.tanh()).abs() < 1e-5);
        assert!((out[2] - 4.0f32.tanh()).abs() < 1e-5);
    }

    #[test]
    fn test_embed_to_stimuli_with_width_downsamples() {
        let t = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0], &[8]);
        let out = embed_to_stimuli_with_width(&t, 4);
        assert_eq!(out.len(), 4);
        assert!((out[0] - 0.0f32.tanh()).abs() < 1e-5);
        assert!((out[1] - 1.0f32.tanh()).abs() < 1e-5);
        assert!((out[2] - 2.0f32.tanh()).abs() < 1e-5);
        assert!((out[3] - 3.0f32.tanh()).abs() < 1e-5);
    }

    #[test]
    fn test_embed_to_stimuli_with_width_pads() {
        let t = Tensor::from_vec(vec![0.5, -0.5], &[2]);
        let out = embed_to_stimuli_with_width(&t, 5);
        assert_eq!(out.len(), 5);
        assert!((out[0] - 0.5f32.tanh()).abs() < 1e-5);
        assert!((out[1] - (-0.5f32).tanh()).abs() < 1e-5);
        for v in &out[2..] {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_embed_to_stimuli_with_width_strictly_bounded_after_resize() {
        let t = Tensor::from_vec(vec![500.0; 128], &[128]);
        let out = embed_to_stimuli_with_width(&t, 16);
        assert_eq!(out.len(), 16);
        for v in &out {
            assert!(v.abs() <= 1.0, "tanh bound violated: {v}");
            assert!(*v > 0.99);
        }
    }

    #[test]
    fn test_empty_width_returns_empty() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let out = embed_to_stimuli_with_width(&t, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn rate_sum_projects_to_embed_dim_and_bounds() {
        let act = SpikeActivity::from_fired(&[0, 2], 4).unwrap();
        let emb = project_spike_activity(ProjectionMode::RateSum, &act, 4, 8).unwrap();
        assert_eq!(emb.len(), 8);
        for v in &emb {
            assert!(*v >= -1.0 && *v <= 1.0, "tanh bound, got {v}");
        }
    }

    #[test]
    fn pure_modes_run_without_backends() {
        let mut act = SpikeActivity::from_fired(&[1], 3).unwrap();
        act.potentials = vec![0.1, 0.5, 0.9];
        for mode in [
            ProjectionMode::RateSum,
            ProjectionMode::TemporalHistogram,
            ProjectionMode::MembraneSnapshot,
            ProjectionMode::SpikingTernary,
        ] {
            let emb = project_spike_activity(mode, &act, 3, 4).unwrap();
            assert_eq!(emb.len(), 4, "mode {mode:?}");
        }
    }

    #[test]
    fn spiking_ternary_features_match_rate_sum() {
        let act = SpikeActivity::from_fired(&[0], 2).unwrap();
        let a = spike_activity_features(ProjectionMode::RateSum, &act, 2).unwrap();
        let b = spike_activity_features(ProjectionMode::SpikingTernary, &act, 2).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn project_rejects_zero_embed_dim() {
        let act = SpikeActivity::from_fired(&[], 2).unwrap();
        assert!(project_spike_activity(ProjectionMode::RateSum, &act, 2, 0).is_err());
    }

    #[test]
    fn project_rejects_short_potentials() {
        let act = SpikeActivity {
            spike_train: vec![vec![]],
            potentials: vec![0.0],
            iz_potentials: vec![],
        };
        match project_spike_activity(ProjectionMode::RateSum, &act, 4, 4) {
            Err(HybridError::InputLengthMismatch {
                expected: 4,
                got: 1,
            }) => {}
            other => panic!("unexpected {other:?}"),
        }
    }
}
