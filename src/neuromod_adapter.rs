// SPDX-License-Identifier: MIT OR Apache-2.0

//! Real SNN backend adapter over [`neuromod`](https://crates.io/crates/neuromod) 0.7.
//!
//! [`NeuromodSnn`] implements hybrid-fusion's [`crate::SpikingNetwork`] trait on
//! top of `neuromod::SpikingNetwork`, a genuine spiking engine with LIF and
//! Izhikevich neuron dynamics. It is the *real* SNN backend for this crate; the
//! `SimpleSnn` reference implementation (behind the `backends` feature) is only a
//! deterministic mock for tests and examples.
//!
//! This module is compiled only when the optional `neuromod` feature is enabled.
//!
//! # Dependency-boundary exception
//!
//! `REVIEW.md` forbids concrete backend dependencies in this crate, and
//! `AGENTS.md` permits one only as a tracked, documented exception. The optional
//! `neuromod` dependency behind this module is exactly that exception, authorized
//! by [issue #44](https://github.com/rmems/hybrid-fusion/issues/44). It is
//! feature-gated and off by default, so the crate's default build stays
//! trait-only and backend-agnostic. The boundary is preserved: no `neuromod` type
//! appears in any public signature or re-export (see below), and downstream
//! crates remain the long-term home for concrete backends.
//!
//! # Public boundary
//!
//! No `neuromod` type ever appears in a public signature or re-export of
//! hybrid-fusion. Callers work exclusively with hybrid-fusion vocabulary:
//! [`crate::NeuroModulators`] in, [`crate::Result<Vec<usize>>`](crate::Result)
//! out, and [`crate::HybridError`] on failure. The conversion to neuromod's own
//! `NeuroModulators` and the mapping of `neuromod::StepError` to
//! [`crate::HybridError`] happen internally.
//!
//! # Neuromodulator mapping
//!
//! hybrid-fusion's [`crate::NeuroModulators`] has a different vocabulary from
//! neuromod's. The adapter converts explicitly, setting every neuromod field
//! (no implicit field reuse), in `to_neuromod_modulators`:
//!
//! | hybrid-fusion field | neuromod field   | rule                                            |
//! |---------------------|------------------|-------------------------------------------------|
//! | `dopamine`          | `dopamine`       | `(dopamine + aux_dopamine).clamp(0.0, 1.0)`      |
//! | `aux_dopamine`      | `dopamine`       | folded into `dopamine` (secondary reward channel)|
//! | `cortisol`          | `norepinephrine` | direct (`norepinephrine = cortisol`); NE models stress/arousal |
//! | `acetylcholine`     | `acetylcholine`  | direct                                          |
//! | `tempo`             | *(none)*         | intentionally **not** mapped (see below)        |
//! | *(none)*            | `serotonin`      | left at `0.0` (no hybrid-fusion source)          |
//!
//! `tempo` is a step/timebase policy in hybrid-fusion. neuromod's `step` takes no
//! timebase input, so there is no neuromod modulator to carry it; it is
//! intentionally dropped from the conversion rather than forced onto an unrelated
//! channel. `serotonin` has no hybrid-fusion source and is left at its neuromod
//! default of `0.0`.
//!
//! Only the folded `dopamine` channel is range-clamped to `[0, 1]`; the
//! `norepinephrine` and `acetylcholine` channels are forwarded as-is and rely on
//! neuromod's finiteness check rather than a range check.
//!
//! # Seeded-replay contract
//!
//! [`NeuromodSnn`] owns a `neuromod::StdRng`. The [`SpikingNetwork::step`] path
//! routes through `neuromod::SpikingNetwork::step_with_rng` using that owned
//! generator, so the adapter is deterministic *by construction*: two
//! [`NeuromodSnn::with_seed`] instances built with the same seed and fed the same
//! stimulus sequence produce identical fired-index sequences. [`NeuromodSnn::reset`]
//! resets the neuron dynamics but does **not** reseed the RNG; use
//! [`NeuromodSnn::reseed`] to restart the random stream explicitly. A caller who
//! wants nondeterminism can [`reseed`](NeuromodSnn::reseed) from an entropy source.

use crate::error::{HybridError, Result};
use crate::traits::{NeuroModulators, SpikingNetwork};
use neuromod::{SeedableRng, StdRng};

/// Default RNG seed used by [`NeuromodSnn::new`] when the caller does not supply
/// one. [`NeuromodSnn::with_seed`] overrides this for explicit replay.
const DEFAULT_SEED: u64 = 0x4859_4252_4944_5F53; // "HYBRID_S"

/// Default number of leaky-integrate-and-fire neurons allocated per network.
///
/// The neuromod engine sizes its LIF/Izhikevich banks independently of the
/// input channel count. We pick small, non-zero defaults so a plain
/// [`NeuromodSnn::new`] produces a functional network; callers needing a
/// specific topology can build the network elsewhere and wrap it with
/// [`NeuromodSnn::from_network`].
const DEFAULT_NUM_LIF: usize = 8;

/// Default number of Izhikevich neurons allocated per network. See
/// [`DEFAULT_NUM_LIF`] for the rationale behind fixed small defaults.
const DEFAULT_NUM_IZH: usize = 1;

/// Initial positive synaptic weight seeded onto every LIF input channel at
/// construction.
///
/// `neuromod::SpikingNetwork::with_dimensions` initializes every LIF synaptic
/// weight to `0.0`, and its `step` integrates *weighted* stimuli. Connectivity
/// only grows via R-STDP, which requires a first post-synaptic spike — so with
/// all-zero weights that first spike never occurs and the backend can never
/// fire. We seed a small, uniform positive weight so a plain
/// [`NeuromodSnn::new`] / [`NeuromodSnn::with_seed`] produces a network that
/// actually spikes under stimulus. The value is a fixed constant (not an RNG
/// draw) to keep construction trivially reproducible; the owned seeded RNG still
/// drives the stochastic step dynamics, so determinism-by-construction holds.
const INITIAL_LIF_WEIGHT: f32 = 0.5;

/// Real SNN backend adapter over `neuromod::SpikingNetwork`.
///
/// Wraps a neuromod spiking network plus an owned seeded RNG so stepping is
/// deterministic by construction. Implements [`crate::SpikingNetwork`]. See the
/// [module documentation](self) for the neuromodulator mapping and seeded-replay
/// contract.
pub struct NeuromodSnn {
    inner: neuromod::SpikingNetwork,
    rng: StdRng,
}

impl NeuromodSnn {
    /// Build an adapter with `num_channels` input channels and small default
    /// neuron banks (`num_lif = 8`, `num_izh = 1`), seeded from a fixed default
    /// seed.
    ///
    /// For deterministic replay you control, prefer [`Self::with_seed`]. As with
    /// [`Self::with_seed`], passing `num_channels == 0` yields a backend that
    /// [`HybridNetwork::try_new`](crate::HybridNetwork::try_new) will reject.
    pub fn new(num_channels: usize) -> Self {
        Self::with_seed(num_channels, DEFAULT_SEED)
    }

    /// Build an adapter with `num_channels` input channels, small default neuron
    /// banks (`num_lif = 8`, `num_izh = 1`), and an RNG seeded from `seed`.
    ///
    /// This is the deterministic-replay entry point: two instances built with the
    /// same `seed` and stepped over the same stimulus sequence emit identical
    /// fired-index sequences.
    ///
    /// Each LIF input synapse is seeded with a small positive weight
    /// ([`INITIAL_LIF_WEIGHT`]) so the network can actually fire under stimulus:
    /// neuromod initializes all synaptic weights to `0.0` and only grows
    /// connectivity via R-STDP after a first post-synaptic spike, which never
    /// occurs from an all-zero weight matrix.
    ///
    /// # Zero channels
    ///
    /// Passing `num_channels == 0` builds a backend whose
    /// [`num_channels()`](SpikingNetwork::num_channels) is `0`. The constructor is
    /// infallible and does not panic in release, but the `SpikingNetwork` contract
    /// requires at least one channel, so
    /// [`HybridNetwork::try_new`](crate::HybridNetwork::try_new) (and `forward`)
    /// will reject such a backend downstream. A `debug_assert!` flags the misuse
    /// early in debug builds.
    pub fn with_seed(num_channels: usize, seed: u64) -> Self {
        debug_assert!(
            num_channels > 0,
            "NeuromodSnn requires num_channels > 0; a zero-channel backend is \
             rejected by HybridNetwork::try_new/forward"
        );
        let mut inner = neuromod::SpikingNetwork::with_dimensions(
            DEFAULT_NUM_LIF,
            DEFAULT_NUM_IZH,
            num_channels,
        );
        // Seed positive input weights so the LIF bank can spike. Only weight
        // *values* change; the per-neuron `weights` length (and the matching
        // `eligibility` length) are left as neuromod built them.
        for neuron in &mut inner.neurons {
            neuron.weights.fill(INITIAL_LIF_WEIGHT);
        }
        Self {
            inner,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Wrap a caller-constructed neuromod network with an RNG seeded from `seed`.
    ///
    /// Kept `pub(crate)` so no `neuromod` type leaks across the public boundary;
    /// used by tests that need a bespoke topology.
    #[allow(dead_code)]
    pub(crate) fn from_network(inner: neuromod::SpikingNetwork, seed: u64) -> Self {
        Self {
            inner,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Reset the underlying neuron dynamics (membranes, spikes, counters) to
    /// their initial state.
    ///
    /// This does **not** reseed the RNG: replay is explicit. To restart the
    /// random stream, call [`Self::reseed`]. Resetting dynamics without reseeding
    /// lets a caller continue a single random stream across logical episodes.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Reseed the owned RNG from `seed`, restarting the deterministic stream.
    ///
    /// Combine with [`Self::reset`] to replay a run from a known starting point.
    pub fn reseed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }

    /// Convert hybrid-fusion [`NeuroModulators`] into neuromod's own
    /// `NeuroModulators`, setting every neuromod field explicitly.
    ///
    /// Mapping (see the [module documentation](self) for the rationale):
    ///
    /// - `dopamine` <- `(m.dopamine + m.aux_dopamine).clamp(0.0, 1.0)` — the
    ///   secondary reward channel `aux_dopamine` is folded additively into
    ///   dopamine, then clamped into `[0, 1]`.
    /// - `norepinephrine` <- `m.cortisol` — NE models stress/arousal, the
    ///   closest neuromod analogue to hybrid-fusion cortisol.
    /// - `acetylcholine` <- `m.acetylcholine` — direct.
    /// - `serotonin` <- `0.0` — no hybrid-fusion source; neuromod default.
    /// - `m.tempo` is intentionally **not** mapped: it is a step/timebase policy
    ///   and neuromod's `step` has no timebase input.
    ///
    /// Only the folded `dopamine` channel is range-clamped to `[0, 1]`; the
    /// `norepinephrine` (from `cortisol`) and `acetylcholine` channels are
    /// forwarded as-is and rely on neuromod's finiteness check rather than a
    /// range check.
    fn to_neuromod_modulators(m: &NeuroModulators) -> neuromod::NeuroModulators {
        neuromod::NeuroModulators {
            dopamine: (m.dopamine + m.aux_dopamine).clamp(0.0, 1.0),
            serotonin: 0.0,
            acetylcholine: m.acetylcholine,
            norepinephrine: m.cortisol,
        }
    }
}

/// Convert a `neuromod::StepError` into a [`HybridError`].
///
/// The match is exhaustive with no wildcard: `neuromod::StepError` is **not**
/// `#[non_exhaustive]`, so every one of its six 0.7 variants is named here. This
/// is intentional — a future neuromod release that adds a variant will fail to
/// compile until the mapping is updated, rather than silently falling through a
/// catch-all.
///
/// `InputLenMismatch` maps to [`HybridError::InputLengthMismatch`] (structured,
/// with `expected`/`got`); every other variant carries its `Display` text into
/// [`HybridError::SnnStep`].
fn map_step_error(e: neuromod::StepError) -> HybridError {
    use neuromod::StepError as E;
    match e {
        E::InputLenMismatch { expected, got } => HybridError::InputLengthMismatch { expected, got },
        E::NonFiniteStimulus { .. }
        | E::NonFiniteModulator { .. }
        | E::NonFinitePredictiveState { .. }
        | E::StepCounterExhausted { .. }
        | E::CheckpointShapeMismatch { .. } => HybridError::SnnStep(e.to_string()),
    }
}

impl SpikingNetwork for NeuromodSnn {
    /// Advance one spiking step.
    ///
    /// Routes through the owned seeded RNG (`neuromod`'s `step_with_rng`) so the
    /// adapter is deterministic by construction. Converts modulators via the
    /// internal `to_neuromod_modulators` helper and maps any `neuromod::StepError`
    /// through the internal `map_step_error` helper.
    fn step(&mut self, stimuli: &[f32], modulators: &NeuroModulators) -> Result<Vec<usize>> {
        let m = Self::to_neuromod_modulators(modulators);
        self.inner
            .step_with_rng(stimuli, &m, &mut self.rng)
            .map_err(map_step_error)
    }

    fn num_channels(&self) -> usize {
        // `num_channels` is a public field on neuromod::SpikingNetwork, not a method.
        self.inner.num_channels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 4;
    const SEED: u64 = 0xC0FF_EE01;

    // ── Per-field modulator mapping ────────────────────────────────────────

    #[test]
    fn maps_dopamine_with_aux_folded_in() {
        let m = NeuroModulators {
            dopamine: 0.3,
            aux_dopamine: 0.2,
            cortisol: 0.0,
            acetylcholine: 0.0,
            tempo: 1.0,
        };
        let out = NeuromodSnn::to_neuromod_modulators(&m);
        assert!((out.dopamine - 0.5).abs() < 1e-6, "dopamine = da + aux");
    }

    #[test]
    fn clamps_folded_dopamine_to_unit_interval() {
        let m = NeuroModulators {
            dopamine: 0.8,
            aux_dopamine: 0.5,
            cortisol: 0.0,
            acetylcholine: 0.0,
            tempo: 1.0,
        };
        let out = NeuromodSnn::to_neuromod_modulators(&m);
        assert!((out.dopamine - 1.0).abs() < 1e-6, "0.8 + 0.5 clamps to 1.0");

        let m_low = NeuroModulators {
            dopamine: -1.0,
            aux_dopamine: 0.2,
            ..Default::default()
        };
        let out_low = NeuromodSnn::to_neuromod_modulators(&m_low);
        assert!(out_low.dopamine >= 0.0, "clamps to lower bound 0.0");
    }

    #[test]
    fn maps_cortisol_to_norepinephrine() {
        let m = NeuroModulators {
            cortisol: 0.42,
            ..Default::default()
        };
        let out = NeuromodSnn::to_neuromod_modulators(&m);
        assert!((out.norepinephrine - 0.42).abs() < 1e-6);
    }

    #[test]
    fn maps_acetylcholine_directly() {
        let m = NeuroModulators {
            acetylcholine: 0.37,
            ..Default::default()
        };
        let out = NeuromodSnn::to_neuromod_modulators(&m);
        assert!((out.acetylcholine - 0.37).abs() < 1e-6);
    }

    #[test]
    fn serotonin_stays_zero() {
        // No hybrid-fusion field maps to serotonin; it stays at the 0.0 default
        // regardless of the source modulators.
        let m = NeuroModulators {
            dopamine: 1.0,
            aux_dopamine: 1.0,
            cortisol: 1.0,
            acetylcholine: 1.0,
            tempo: 5.0,
        };
        let out = NeuromodSnn::to_neuromod_modulators(&m);
        assert_eq!(out.serotonin, 0.0);
    }

    #[test]
    fn tempo_does_not_appear_in_neuromod_modulators() {
        // neuromod::NeuroModulators has exactly {dopamine, serotonin,
        // acetylcholine, norepinephrine}. Varying only `tempo` must not change
        // any neuromod field: tempo is intentionally unmapped.
        let base = NeuroModulators {
            dopamine: 0.2,
            aux_dopamine: 0.1,
            cortisol: 0.3,
            acetylcholine: 0.4,
            tempo: 1.0,
        };
        let bumped = NeuroModulators {
            tempo: 99.0,
            ..base.clone()
        };
        let a = NeuromodSnn::to_neuromod_modulators(&base);
        let b = NeuromodSnn::to_neuromod_modulators(&bumped);
        assert_eq!(a.dopamine, b.dopamine);
        assert_eq!(a.serotonin, b.serotonin);
        assert_eq!(a.acetylcholine, b.acetylcholine);
        assert_eq!(a.norepinephrine, b.norepinephrine);
    }

    // ── Shape / channels ───────────────────────────────────────────────────

    #[test]
    fn num_channels_matches_constructor_arg() {
        let snn = NeuromodSnn::new(N);
        assert_eq!(snn.num_channels(), N);
    }

    #[test]
    fn step_returns_indices_within_channel_range() {
        let mut snn = NeuromodSnn::with_seed(N, SEED);
        let mods = NeuroModulators::default();
        for _ in 0..16 {
            let fired = snn.step(&[0.9; N], &mods).unwrap();
            // Fired LIF indices are bounded by the LIF bank size (8), not the
            // channel count; assert the tighter, always-true channel bound plus
            // the neuron-bank bound so the check stays meaningful once firing
            // actually happens.
            assert!(
                fired.iter().all(|&i| i < DEFAULT_NUM_LIF),
                "fired indices must be < num_lif ({DEFAULT_NUM_LIF})"
            );
        }
    }

    #[test]
    fn backend_actually_fires_under_stimulus() {
        // Regression for the all-zero-weight bug: neuromod initializes every LIF
        // synaptic weight to 0.0, so before the constructor seeds positive
        // weights the backend never spikes. Assert that a fresh network emits at
        // least one non-empty fired-index vector over a short run.
        let mut snn = NeuromodSnn::with_seed(N, SEED);
        let mods = NeuroModulators::default();
        let fired_any = (0..32).any(|_| {
            let fired = snn.step(&[0.9; N], &mods).unwrap();
            assert!(
                fired.iter().all(|&i| i < DEFAULT_NUM_LIF),
                "fired indices must be < num_lif ({DEFAULT_NUM_LIF})"
            );
            !fired.is_empty()
        });
        assert!(
            fired_any,
            "seeded NeuromodSnn must fire at least once under stimulus"
        );
    }

    // ── Zero-channel construction (documented degenerate case) ─────────────

    // A zero-channel network is a documented degenerate case. Constructors are
    // infallible: in release builds they build a backend that reports
    // `num_channels() == 0` (which HybridNetwork::try_new/forward then reject),
    // while in debug builds a `debug_assert!` flags the misuse eagerly. Pin both
    // halves of that contract so neither regresses.
    #[cfg(not(debug_assertions))]
    #[test]
    fn zero_channels_constructs_without_panic_in_release() {
        let snn = NeuromodSnn::new(0);
        assert_eq!(
            snn.num_channels(),
            0,
            "release build constructs a zero-channel backend without panicking"
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "num_channels > 0")]
    fn zero_channels_debug_asserts() {
        let _ = NeuromodSnn::new(0);
    }

    // ── Width validation ───────────────────────────────────────────────────

    #[test]
    fn rejects_wrong_width_with_input_length_mismatch() {
        let mut snn = NeuromodSnn::with_seed(N, SEED);
        let mods = NeuroModulators::default();
        let err = snn.step(&[0.1, 0.2], &mods).unwrap_err();
        match err {
            HybridError::InputLengthMismatch { expected, got } => {
                assert_eq!(expected, N);
                assert_eq!(got, 2);
            }
            other => panic!("expected InputLengthMismatch, got {other:?}"),
        }
    }

    // ── Finiteness / error mapping ─────────────────────────────────────────

    #[test]
    fn nan_stimulus_maps_to_snn_step_error_not_panic() {
        let mut snn = NeuromodSnn::with_seed(N, SEED);
        let mods = NeuroModulators::default();
        let mut stim = [0.5; N];
        stim[1] = f32::NAN;
        let err = snn.step(&stim, &mods).unwrap_err();
        assert!(
            matches!(err, HybridError::SnnStep(_)),
            "NaN stimulus must map to SnnStep, got {err:?}"
        );
    }

    // ── Deterministic replay ───────────────────────────────────────────────

    fn stimulus_sequence() -> Vec<[f32; N]> {
        vec![
            [0.9, 0.1, 0.5, 0.3],
            [0.2, 0.8, 0.4, 0.6],
            [0.7, 0.7, 0.1, 0.9],
            [0.3, 0.5, 0.9, 0.2],
        ]
    }

    #[test]
    fn same_seed_produces_identical_fired_sequences() {
        let mods = NeuroModulators::default();
        let run = |seed: u64| {
            let mut snn = NeuromodSnn::with_seed(N, seed);
            stimulus_sequence()
                .iter()
                .map(|s| snn.step(s, &mods).unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(run(SEED), run(SEED), "same seed => same fired indices");
    }

    #[test]
    fn reset_then_replay_is_reproducible() {
        let mods = NeuroModulators::default();
        let mut snn = NeuromodSnn::with_seed(N, SEED);
        let first: Vec<_> = stimulus_sequence()
            .iter()
            .map(|s| snn.step(s, &mods).unwrap())
            .collect();

        // Reset dynamics and reseed to the same seed to replay from scratch.
        snn.reset();
        snn.reseed(SEED);
        let second: Vec<_> = stimulus_sequence()
            .iter()
            .map(|s| snn.step(s, &mods).unwrap())
            .collect();

        assert_eq!(first, second, "reset + reseed replays identically");
    }

    #[test]
    fn map_step_error_maps_input_len_mismatch_structurally() {
        let mapped = map_step_error(neuromod::StepError::InputLenMismatch {
            expected: 7,
            got: 3,
        });
        assert!(matches!(
            mapped,
            HybridError::InputLengthMismatch {
                expected: 7,
                got: 3
            }
        ));
    }
}
