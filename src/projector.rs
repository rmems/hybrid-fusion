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
/// Each [`ProjectionMode`] emits a **distinct** pure representation (issue #25):
/// - [`RateSum`](ProjectionMode::RateSum) / [`SpikingTernary`](ProjectionMode::SpikingTernary):
///   per-neuron firing rates only (`len == n_neurons`)
/// - [`TemporalHistogram`](ProjectionMode::TemporalHistogram): time-binned rates only
///   (`len == n_neurons * 4`)
/// - [`MembraneSnapshot`](ProjectionMode::MembraneSnapshot): clamped membranes only
///   (`len == n_neurons`)
///
/// Unlike corinth-canal’s full rates+hist+membrane+iz concat (for a learned
/// linear layer), pure hybrid-fusion modes do not share a common concatenation.
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
    if activity.potentials.len() != n_neurons {
        return Err(HybridError::InputLengthMismatch {
            expected: n_neurons,
            got: activity.potentials.len(),
        });
    }
    for (i, &v) in activity.potentials.iter().enumerate() {
        if !v.is_finite() {
            return Err(HybridError::InvalidConfig(format!(
                "spike_activity_features: non-finite potential at index {i}"
            )));
        }
    }
    for (t, step) in activity.spike_train.iter().enumerate() {
        if let Some(&idx) = step.iter().find(|&&i| i >= n_neurons) {
            return Err(HybridError::InvalidConfig(format!(
                "spike_activity_features: spike index {idx} out of range at step {t} (n_neurons={n_neurons})"
            )));
        }
    }

    match mode {
        ProjectionMode::RateSum | ProjectionMode::SpikingTernary => {
            // SpikingTernary pure path matches RateSum; GIF lives in neuromod.
            Ok(firing_rates(activity, n_neurons))
        }
        ProjectionMode::TemporalHistogram => Ok(temporal_histogram(activity, n_neurons)),
        ProjectionMode::MembraneSnapshot => Ok(activity
            .potentials
            .iter()
            .map(|&v| v.clamp(0.0, 1.0))
            .collect()),
    }
}

/// Per-neuron mean spike counts over timesteps (`len == n_neurons`).
fn firing_rates(activity: &SpikeActivity, n_neurons: usize) -> Vec<f32> {
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
    rates
}

/// Time-binned spike rates (`len == n_neurons * TEMPORAL_BINS`), each bin
/// normalized by its actual timestep width.
fn temporal_histogram(activity: &SpikeActivity, n_neurons: usize) -> Vec<f32> {
    let bins = TEMPORAL_BINS;
    let mut hist = vec![0.0_f32; n_neurons * bins];
    if activity.spike_train.is_empty() {
        return hist;
    }
    let steps = activity.spike_train.len();
    let mut bin_steps = vec![0_usize; bins];
    for (t, step) in activity.spike_train.iter().enumerate() {
        let bin = ((t * bins) / steps).min(bins - 1);
        bin_steps[bin] += 1;
        for &idx in step {
            hist[idx * bins + bin] += 1.0;
        }
    }
    for neuron in 0..n_neurons {
        for bin in 0..bins {
            let width = bin_steps[bin];
            if width > 0 {
                hist[neuron * bins + bin] /= width as f32;
            }
        }
    }
    hist
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

    #[test]
    fn project_rejects_long_potentials() {
        let act = SpikeActivity {
            spike_train: vec![vec![]],
            potentials: vec![0.0; 5],
            iz_potentials: vec![],
        };
        match project_spike_activity(ProjectionMode::RateSum, &act, 4, 4) {
            Err(HybridError::InputLengthMismatch {
                expected: 4,
                got: 5,
            }) => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn project_rejects_non_finite_potentials() {
        let act = SpikeActivity {
            spike_train: vec![vec![]],
            potentials: vec![0.0, f32::NAN],
            iz_potentials: vec![],
        };
        match project_spike_activity(ProjectionMode::RateSum, &act, 2, 4) {
            Err(HybridError::InvalidConfig(msg)) => assert!(msg.contains("non-finite")),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn histogram_five_steps_per_bin_normalized() {
        // 5 steps → bin widths [2,1,1,1]; each bin gets one fire of neuron 0.
        // After per-bin normalize, each non-empty bin should be 1.0 for that neuron.
        let act = SpikeActivity {
            spike_train: vec![vec![0], vec![0], vec![0], vec![0], vec![0]],
            potentials: vec![0.0; 2],
            iz_potentials: vec![],
        };
        let feats = spike_activity_features(ProjectionMode::TemporalHistogram, &act, 2).unwrap();
        assert_eq!(feats.len(), 2 * 4);
        // neuron 0 hist occupies [0..4]
        for (bin, &v) in feats.iter().take(4).enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "bin {bin} expected ~1.0 after per-width norm, got {v}"
            );
        }
    }

    #[test]
    fn modes_emit_distinct_feature_dims_and_content() {
        let act = SpikeActivity {
            spike_train: vec![vec![0], vec![1], vec![0]],
            potentials: vec![0.25, 0.75],
            iz_potentials: vec![],
        };
        let rates = spike_activity_features(ProjectionMode::RateSum, &act, 2).unwrap();
        let hist = spike_activity_features(ProjectionMode::TemporalHistogram, &act, 2).unwrap();
        let mem = spike_activity_features(ProjectionMode::MembraneSnapshot, &act, 2).unwrap();
        let tern = spike_activity_features(ProjectionMode::SpikingTernary, &act, 2).unwrap();

        assert_eq!(rates.len(), 2);
        assert_eq!(hist.len(), 8);
        assert_eq!(mem.len(), 2);
        assert_eq!(tern, rates);

        // Membrane is potentials only (clamped), not spike rates.
        assert!((mem[0] - 0.25).abs() < 1e-5);
        assert!((mem[1] - 0.75).abs() < 1e-5);
        // RateSum: neuron0 fires 2/3, neuron1 fires 1/3
        assert!((rates[0] - 2.0 / 3.0).abs() < 1e-5);
        assert!((rates[1] - 1.0 / 3.0).abs() < 1e-5);
        assert_ne!(rates, mem);
    }
}
