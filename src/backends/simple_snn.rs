// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deterministic **mock / reference** spiking network (Leaky Integrate-and-Fire).
//!
//! `SimpleSnn` is a lightweight, fully deterministic reference implementation of
//! the [`SpikingNetwork`] trait intended for tests, examples, and documentation.
//! It is **not** a real neuron-dynamics engine: it models a single leaky
//! integrator per channel with a fixed threshold and no stochastic input
//! encoding, subthreshold dynamics, or plasticity.
//!
//! The **real** SNN backend is `NeuromodSnn`, available behind the optional
//! `neuromod` feature; it wraps the `neuromod` 0.7 spiking engine
//! (LIF/Izhikevich dynamics). Prefer `NeuromodSnn` for production stacks and use
//! `SimpleSnn` only where a small, dependency-free, deterministic stand-in is
//! wanted.
//!
//! Each step:
//! 1. Leak the membrane toward zero by `leak_factor`.
//! 2. Add stimulus, scaled by dopamine (excitatory gain) and dampened by
//!    cortisol (inhibitory gain).
//! 3. If potential exceeds `threshold`, record the neuron as fired and reset
//!    its potential.

use crate::error::{HybridError, Result};
use crate::traits::{NeuroModulators, SpikingNetwork};

/// Deterministic mock / reference Leaky Integrate-and-Fire spiking network.
///
/// This is a reference implementation for tests and examples, not a real
/// dynamics engine. For production use enable the `neuromod` feature and use
/// `NeuromodSnn`, the real SNN backend.
pub struct SimpleSnn {
    membrane_potentials: Vec<f32>,
    threshold: f32,
    leak_factor: f32,
    num_channels: usize,
}

impl SimpleSnn {
    /// Create a new SNN with `num_channels` neurons and sensible defaults
    /// (threshold = 1.0, leak = 0.2).
    pub fn new(num_channels: usize) -> Self {
        Self::with_params(num_channels, 1.0, 0.2)
    }

    /// Create with explicit threshold and leak factor.
    pub fn with_params(num_channels: usize, threshold: f32, leak_factor: f32) -> Self {
        Self {
            membrane_potentials: vec![0.0; num_channels],
            threshold,
            leak_factor: leak_factor.clamp(0.0, 1.0),
            num_channels,
        }
    }

    /// Reset all membrane potentials to zero.
    pub fn reset(&mut self) {
        self.membrane_potentials.fill(0.0);
    }
}

impl SpikingNetwork for SimpleSnn {
    fn step(&mut self, stimuli: &[f32], modulators: &NeuroModulators) -> Result<Vec<usize>> {
        // Reject mismatched widths before mutating membrane state so malformed
        // backend calls cannot silently drop or ignore channels.
        if stimuli.len() != self.num_channels {
            return Err(HybridError::InputLengthMismatch {
                expected: self.num_channels,
                got: stimuli.len(),
            });
        }

        let mut fired = Vec::new();

        let dopamine_gain = 1.0 + modulators.dopamine * 0.5 + modulators.aux_dopamine * 0.25;
        let cortisol_suppress = (1.0 - modulators.cortisol * 0.3).max(0.1);
        let ach_leak_scale = 1.0 + modulators.acetylcholine * 0.3;
        // Clamp tempo to non-negative so leak decays (never amplifies) membrane.
        let tempo = modulators.tempo.max(0.0);
        let effective_leak = (self.leak_factor * ach_leak_scale * tempo).clamp(0.0, 1.0);

        for (i, pot) in self.membrane_potentials.iter_mut().enumerate() {
            // 1. Leak toward zero.
            *pot *= 1.0 - effective_leak;

            // 2. Add stimulus with modulator scaling (width already validated).
            *pot += stimuli[i] * dopamine_gain * cortisol_suppress;

            // 3. Spike if above threshold.
            if *pot >= self.threshold {
                fired.push(i);
                *pot = 0.0;
            }
        }

        Ok(fired)
    }

    fn num_channels(&self) -> usize {
        self.num_channels
    }

    fn capabilities(&self) -> crate::BackendCapabilities {
        // Membrane potentials persist across steps. The trait default leaves
        // `stateful` false so a stimuli-only backend is not treated as
        // recurrent. Reset stays unadvertised: it is a concrete method, not a
        // trait operation.
        use crate::capabilities::RequiredFeature;
        crate::BackendCapabilities::snn(std::any::type_name::<Self>())
            .with_dtypes([crate::Dtype::F32])
            .with_channels(self.num_channels())
            .with_num_neurons(self.num_neurons())
            .with_features([RequiredFeature::new("step")])
            .with_stateful(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fires_when_above_threshold() {
        let mut snn = SimpleSnn::with_params(4, 1.0, 0.0);
        let fired = snn
            .step(&[1.5, 0.0, 0.0, 0.0], &NeuroModulators::default())
            .unwrap();
        assert_eq!(fired, vec![0usize]);
    }

    #[test]
    fn no_fire_below_threshold() {
        let mut snn = SimpleSnn::with_params(4, 1.0, 0.0);
        let fired = snn
            .step(&[0.5, 0.0, 0.0, 0.0], &NeuroModulators::default())
            .unwrap();
        assert!(fired.is_empty());
    }

    #[test]
    fn negative_tempo_does_not_amplify_membrane() {
        let mut snn = SimpleSnn::with_params(1, 10.0, 0.5);
        let mods = NeuroModulators {
            tempo: -2.0,
            ..Default::default()
        };
        snn.step(&[0.4], &mods).unwrap();
        // After leak with clamped tempo=0, potential stays at stimulus (no amp).
        // A second step with zero stimulus and negative tempo must not grow it.
        snn.step(&[0.0], &mods).unwrap();
        let fired = snn.step(&[0.0], &mods).unwrap();
        assert!(
            fired.is_empty(),
            "membrane must not amplify under neg tempo"
        );
    }

    #[test]
    fn leak_decays_potential() {
        let mut snn = SimpleSnn::with_params(2, 1.0, 1.0);
        snn.step(&[0.8, 0.0], &NeuroModulators::default()).unwrap();
        let fired = snn.step(&[0.0, 0.0], &NeuroModulators::default()).unwrap();
        assert!(fired.is_empty());
    }

    #[test]
    fn dopamine_boosts_excitation() {
        let mut snn = SimpleSnn::with_params(1, 1.0, 0.0);
        let mods = NeuroModulators {
            dopamine: 2.0,
            ..Default::default()
        };
        let fired = snn.step(&[0.8], &mods).unwrap();
        assert_eq!(fired, vec![0usize]);
    }

    #[test]
    fn reset_clears_potentials() {
        let mut snn = SimpleSnn::new(4);
        snn.step(&[0.5; 4], &NeuroModulators::default()).unwrap();
        snn.reset();
        let fired = snn.step(&[0.5; 4], &NeuroModulators::default()).unwrap();
        assert!(fired.is_empty());
    }

    #[test]
    fn accumulates_until_fire() {
        let mut snn = SimpleSnn::with_params(1, 1.0, 0.0);
        let neutral = NeuroModulators {
            dopamine: 0.0,
            cortisol: 0.0,
            acetylcholine: 0.0,
            tempo: 1.0,
            aux_dopamine: 0.0,
        };
        snn.step(&[0.4], &neutral).unwrap();
        snn.step(&[0.4], &neutral).unwrap();
        let fired = snn.step(&[0.3], &neutral).unwrap();
        // 0.4 + 0.4 + 0.3 = 1.1 >= 1.0
        assert_eq!(fired, vec![0usize]);
    }

    #[test]
    fn rejects_stimuli_width_mismatch() {
        let mut snn = SimpleSnn::new(4);
        let mods = NeuroModulators::default();
        let err = snn.step(&[0.1, 0.2], &mods).unwrap_err();
        match err {
            HybridError::InputLengthMismatch { expected, got } => {
                assert_eq!(expected, 4);
                assert_eq!(got, 2);
            }
            other => panic!("expected InputLengthMismatch, got {other:?}"),
        }
        // Over-long inputs are also rejected (no silent truncation).
        let err = snn.step(&[0.0; 5], &mods).unwrap_err();
        assert!(matches!(
            err,
            HybridError::InputLengthMismatch {
                expected: 4,
                got: 5
            }
        ));
    }
}
