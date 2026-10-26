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
/// `n_neurons` is the backend's output population, NOT its input width. Feature
/// position `i` refers to output neuron `i`; resizing to an embedding does not
/// create additional neurons or redistribute fired IDs into input channels.
///
/// Unlike corinth-canal’s full rates+hist+membrane+iz concat (for a learned
/// linear layer), pure hybrid-fusion modes do not share a common concatenation.
///
/// # Input contract (validated before any mode runs)
///
/// - `n_neurons > 0`.
/// - `potentials.len() == n_neurons` and every value finite.
/// - Every `spike_train` step holds in-range indices (`< n_neurons`), and each
///   step is a **set**: a neuron appears at most once. A repeated index in a
///   step is rejected, not collapsed — the forward host already rejects a
///   backend emitting the same neuron twice in one tick
///   ([`HybridError::SnnDuplicateSpike`], RM-2205), so a multiset here cannot
///   be a faithful record of a real tick (RM-1445).
/// - `iz_potentials` is either empty or exactly `n_neurons` long with every
///   value finite. No pure mode reads it (the adaptive bank lives in
///   `neuromod`); a well-formed vector is a documented ignore that is carried
///   for trace fidelity, and any other length is rejected instead of being
///   silently dropped.
///
/// With the set contract enforced, every feature this function returns is
/// bounded: [`RateSum`] rates are per-neuron mean counts over steps in
/// `[0, 1]`, histogram bins in `[0, 1]`, membranes in `[0, 1]`. The `tanh`
/// applied later by [`project_spike_activity`] is a squashing of an already
/// bounded vector, not a substitute for this contract.
///
/// [`RateSum`]: ProjectionMode::RateSum
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
    validate_potential_vectors(activity, n_neurons)?;
    validate_spike_train(activity, n_neurons)?;

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

/// Potential-vector half of the [`spike_activity_features`] input contract.
///
/// `potentials` must be exactly `n_neurons` long and finite. `iz_potentials`,
/// which no pure mode reads, must be empty (the unused spelling) or exactly
/// `n_neurons` long and finite — a documented ignore kept for trace/serde
/// fidelity. Rejecting a malformed length here (rather than ignoring it) is
/// what keeps the "silently ignored 99-element vector" case impossible.
/// Error messages name the public entry point, [`spike_activity_features`].
fn validate_potential_vectors(activity: &SpikeActivity, n_neurons: usize) -> Result<()> {
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
    if !activity.iz_potentials.is_empty() {
        if activity.iz_potentials.len() != n_neurons {
            return Err(HybridError::InputLengthMismatch {
                expected: n_neurons,
                got: activity.iz_potentials.len(),
            });
        }
        for (i, &v) in activity.iz_potentials.iter().enumerate() {
            if !v.is_finite() {
                return Err(HybridError::InvalidConfig(format!(
                    "spike_activity_features: non-finite iz_potential at index {i}"
                )));
            }
        }
    }
    Ok(())
}

/// Spike-train half of the [`spike_activity_features`] input contract.
///
/// Every index must be in range (`< n_neurons`) and every step must be a
/// **set**: a neuron appears at most once. A repeated index is rejected with
/// `InvalidConfig` naming the neuron and step — never collapsed — mirroring
/// the forward host's at-most-once-per-tick rule
/// ([`HybridError::SnnDuplicateSpike`], RM-2205), so a multiset cannot be
/// recorded as a faithful tick (RM-1445). Error messages name the public
/// entry point, [`spike_activity_features`].
fn validate_spike_train(activity: &SpikeActivity, n_neurons: usize) -> Result<()> {
    for (t, step) in activity.spike_train.iter().enumerate() {
        if let Some(&idx) = step.iter().find(|&&i| i >= n_neurons) {
            return Err(HybridError::InvalidConfig(format!(
                "spike_activity_features: spike index {idx} out of range at step {t} (n_neurons={n_neurons})"
            )));
        }
        let mut seen = std::collections::HashSet::with_capacity(step.len());
        for &idx in step {
            if !seen.insert(idx) {
                return Err(HybridError::InvalidConfig(format!(
                    "spike_activity_features: spike index {idx} appears more than once at \
                     step {t} (a timestep is a set; a neuron fires at most once per tick)"
                )));
            }
        }
    }
    Ok(())
}

/// Per-neuron mean spike counts over timesteps (`len == n_neurons`).
///
/// Each returned rate is in `[0, 1]`: with the set contract enforced by
/// [`spike_activity_features`] (a neuron fires at most once per step), a
/// neuron's total count over `T` steps cannot exceed `T`, so the mean over
/// steps cannot exceed `1.0`. Duplicates would break this bound and are
/// rejected upstream — they are not counted here.
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
/// normalized by its actual timestep width. Each bin value is in `[0, 1]`:
/// the set contract (at most one fire per neuron per step) means a neuron
/// cannot contribute more than one count to any bin it appears in.
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

    /// The 2026-09-15 evidence case: three copies of 0 in one step used to
    /// yield rate 3.0 (then `tanh`-saturate). A timestep is a set, so the
    /// duplicate is rejected before any feature is built.
    #[test]
    fn features_reject_duplicate_indices_in_one_step() {
        let act = SpikeActivity {
            spike_train: vec![vec![0, 0, 0]],
            potentials: vec![0.0; 2],
            iz_potentials: vec![],
        };
        // The input contract is mode-independent: every mode is guarded.
        for mode in [
            ProjectionMode::RateSum,
            ProjectionMode::SpikingTernary,
            ProjectionMode::TemporalHistogram,
            ProjectionMode::MembraneSnapshot,
        ] {
            match spike_activity_features(mode, &act, 2) {
                Err(HybridError::InvalidConfig(msg)) => {
                    assert!(
                        msg.contains("index 0 appears more than once at step 0"),
                        "{msg}"
                    );
                }
                other => panic!("mode {mode:?}: unexpected {other:?}"),
            }
        }
    }

    /// The set rule holds for every timestep, not just the first: the
    /// rejection names the step the duplicate sits in.
    #[test]
    fn features_reject_duplicate_indices_in_a_later_step() {
        let act = SpikeActivity {
            spike_train: vec![vec![1], vec![2, 2]],
            potentials: vec![0.0; 3],
            iz_potentials: vec![],
        };
        match spike_activity_features(ProjectionMode::RateSum, &act, 3) {
            Err(HybridError::InvalidConfig(msg)) => {
                assert!(
                    msg.contains("index 2 appears more than once at step 1"),
                    "{msg}"
                );
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    /// `from_fired` builds a one-step record, so it enforces the same set
    /// rule as the projector: the rejection names both the neuron and `step 0`.
    #[test]
    fn from_fired_rejects_duplicate_indices() {
        match SpikeActivity::from_fired(&[0, 1, 0], 2) {
            Err(HybridError::InvalidConfig(msg)) => {
                assert!(
                    msg.contains("index 0 appears more than once at step 0"),
                    "{msg}"
                );
            }
            other => panic!("unexpected {other:?}"),
        }
        // Order / distinctness: an unsorted set is fine, a multiset is not.
        assert!(SpikeActivity::from_fired(&[3, 1, 0], 4).is_ok());
    }

    /// With uniqueness enforced, RateSum rates stay in `[0, 1]` per neuron,
    /// measured as the mean over steps — they cannot exceed 1.0 any more.
    #[test]
    fn rate_sum_rates_are_mean_over_steps_and_bounded() {
        let act = SpikeActivity {
            // neuron 0 fires in 2 of 3 steps, neuron 1 in all 3, neuron 2 never.
            spike_train: vec![vec![0, 1], vec![1], vec![0, 1]],
            potentials: vec![0.0; 3],
            iz_potentials: vec![],
        };
        let rates = spike_activity_features(ProjectionMode::RateSum, &act, 3).unwrap();
        assert_eq!(rates.len(), 3);
        assert!((rates[0] - 2.0 / 3.0).abs() < 1e-6);
        assert!((rates[1] - 1.0).abs() < 1e-6);
        assert!((rates[2] - 0.0).abs() < 1e-6);
        for &r in &rates {
            assert!(
                (0.0..=1.0).contains(&r),
                "set semantics keep per-neuron mean rates in [0, 1], got {r}"
            );
        }
        // A full-population burst on every step saturates at exactly 1.0.
        let full = SpikeActivity {
            spike_train: vec![vec![0, 1, 2]; 7],
            potentials: vec![0.0; 3],
            iz_potentials: vec![],
        };
        let rates = spike_activity_features(ProjectionMode::RateSum, &full, 3).unwrap();
        for &r in &rates {
            assert!(
                (r - 1.0).abs() < 1e-6,
                "full burst rate must be exactly 1.0, got {r}"
            );
        }
    }

    /// `iz_potentials` case 1 of 3: empty is the unused spelling, accepted
    /// by every mode.
    #[test]
    fn iz_potentials_empty_is_accepted_unused() {
        let act = SpikeActivity {
            spike_train: vec![vec![0]],
            potentials: vec![0.0, 0.0],
            iz_potentials: vec![],
        };
        assert_eq!(
            spike_activity_features(ProjectionMode::RateSum, &act, 2).unwrap(),
            vec![1.0, 0.0]
        );
    }

    /// `iz_potentials` case 2 of 3: neuron-sized and finite is a documented
    /// ignore — no pure mode reads it, so features match the empty spelling.
    #[test]
    fn iz_potentials_neuron_sized_is_a_documented_ignore() {
        let empty = SpikeActivity {
            spike_train: vec![vec![0]],
            potentials: vec![0.25, 0.5],
            iz_potentials: vec![],
        };
        let filled = SpikeActivity {
            spike_train: vec![vec![0]],
            potentials: vec![0.25, 0.5],
            iz_potentials: vec![0.9, 0.1],
        };
        for mode in [
            ProjectionMode::RateSum,
            ProjectionMode::SpikingTernary,
            ProjectionMode::TemporalHistogram,
            ProjectionMode::MembraneSnapshot,
        ] {
            let a = spike_activity_features(mode, &empty, 2).unwrap();
            let b = spike_activity_features(mode, &filled, 2).unwrap();
            assert_eq!(a, b, "mode {mode:?} must ignore well-formed iz_potentials");
        }
    }

    /// `iz_potentials` case 3 of 3: any length other than empty or
    /// `n_neurons` is a shape error — the 99-element evidence vector is no
    /// longer silently dropped.
    #[test]
    fn iz_potentials_garbage_length_is_rejected() {
        let act = SpikeActivity {
            spike_train: vec![vec![0]],
            potentials: vec![0.0; 4],
            iz_potentials: vec![0.0; 99],
        };
        match spike_activity_features(ProjectionMode::RateSum, &act, 4) {
            Err(HybridError::InputLengthMismatch {
                expected: 4,
                got: 99,
            }) => {}
            other => panic!("unexpected {other:?}"),
        }
        // Short-but-nonempty is just as wrong.
        let short = SpikeActivity {
            spike_train: vec![vec![0]],
            potentials: vec![0.0; 4],
            iz_potentials: vec![0.0],
        };
        match spike_activity_features(ProjectionMode::MembraneSnapshot, &short, 4) {
            Err(HybridError::InputLengthMismatch {
                expected: 4,
                got: 1,
            }) => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    /// A neuron-sized `iz_potentials` must still be finite: one non-finite
    /// value rejects the whole record rather than poisoning a feature.
    #[test]
    fn iz_potentials_non_finite_is_rejected() {
        let act = SpikeActivity {
            spike_train: vec![vec![0]],
            potentials: vec![0.0, 0.0],
            iz_potentials: vec![0.0, f32::INFINITY],
        };
        match spike_activity_features(ProjectionMode::RateSum, &act, 2) {
            Err(HybridError::InvalidConfig(msg)) => {
                assert!(msg.contains("non-finite iz_potential"), "{msg}");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    /// The contract is enforced at the public `project_spike_activity` entry
    /// point too, not only on the internal feature builder.
    #[test]
    fn iz_potentials_contract_holds_through_project_entry_point() {
        let act = SpikeActivity {
            spike_train: vec![vec![0]],
            potentials: vec![0.0; 2],
            iz_potentials: vec![0.0; 5],
        };
        assert!(project_spike_activity(ProjectionMode::RateSum, &act, 2, 4).is_err());
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
