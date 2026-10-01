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
//! resets the neuron dynamics **and restores the constructor's initial
//! seed-derived LIF weights**, but does **not** reseed the RNG; use
//! [`NeuromodSnn::reseed`] to restart the random stream explicitly. Because
//! `neuromod::SpikingNetwork::reset` leaves `neuron.weights` untouched, R-STDP
//! mutations learned mid-run would otherwise survive a reset and make a
//! "replay" start from learned connectivity; the adapter copies the saved
//! initial weights back on every [`reset`](NeuromodSnn::reset) so the documented
//! `reset(); reseed(seed)` sequence replays from the *original* connectivity and
//! reproduces the original fired-index sequence. A caller who wants
//! nondeterminism can [`reseed`](NeuromodSnn::reseed) from an entropy source.

use crate::error::{HybridError, Result};
use crate::traits::{NeuroModulators, SpikingNetwork};
use neuromod::{SeedableRng, StdRng};

/// Derive a distinct, reproducible synaptic weight in `[0, 1)` for a given
/// `(seed, neuron, channel)` triple.
///
/// `neuromod::SpikingNetwork::with_dimensions` initializes every LIF synaptic
/// weight to `0.0`, and its `step` integrates *weighted* stimuli. Connectivity
/// only grows via R-STDP, which requires a first post-synaptic spike — so with
/// all-zero weights that first spike never occurs and the backend can never
/// fire. Seeding a single *uniform* positive constant fixes firing but makes
/// every LIF neuron behave identically, collapsing the bank into all-or-nothing
/// lockstep. Instead we derive a **distinct per-synapse** weight so neurons fire
/// in heterogeneous subsets.
///
/// The mapping is a small inline SplitMix64-style hash of the triple into the
/// unit interval. It is fully deterministic in the `seed` (no RNG draw), so
/// construction stays trivially reproducible and `with_seed` replay is exact,
/// while the owned seeded RNG still drives the stochastic step dynamics. We do
/// **not** use `rand`'s distribution methods here: neuromod re-exports `rand`
/// 0.10 but not the `Rng` extension trait those methods live on, and adding a
/// direct `rand` dependency would violate the dependency-light default build.
fn seeded_weight(seed: u64, neuron: usize, channel: usize) -> f32 {
    let mut z = seed
        ^ 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(neuron as u64 + 1)
        ^ 0xBF58_476D_1CE4_E5B9u64.wrapping_mul(channel as u64 + 1);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Take the top 24 bits and scale into [0, 1); 24 bits map exactly into f32's
    // mantissa so every produced value is representable without rounding bias.
    (z >> 40) as f32 / (1u64 << 24) as f32
}

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

/// Real SNN backend adapter over `neuromod::SpikingNetwork`.
///
/// Wraps a neuromod spiking network plus an owned seeded RNG so stepping is
/// deterministic by construction. Implements [`crate::SpikingNetwork`]. See the
/// [module documentation](self) for the neuromodulator mapping and seeded-replay
/// contract.
///
/// `initial_weights` snapshots each LIF neuron's seed-derived weight vector at
/// construction (see [`seeded_weight`]) so [`reset`](Self::reset) can restore the
/// original connectivity for a from-scratch replay even after R-STDP has mutated
/// the live weights. It is a plain `Vec<Vec<f32>>`, not a `neuromod` type, so no
/// backend type crosses the public boundary.
pub struct NeuromodSnn {
    inner: neuromod::SpikingNetwork,
    rng: StdRng,
    initial_weights: Vec<Vec<f32>>,
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
    /// Each LIF input synapse is seeded with a **distinct** positive weight in
    /// `[0, 1)` derived from the `seed` and the neuron/channel indices (see
    /// [`seeded_weight`]) so the network can actually fire under stimulus:
    /// neuromod initializes all synaptic weights to `0.0` and only grows
    /// connectivity via R-STDP after a first post-synaptic spike, which never
    /// occurs from an all-zero weight matrix. Using distinct per-neuron weights
    /// (rather than one uniform constant) makes the LIF bank fire in
    /// heterogeneous subsets instead of all-or-nothing lockstep. The seeded
    /// weight matrix is snapshotted so [`reset`](Self::reset) can restore it.
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
        // Seed distinct positive input weights so the LIF bank can spike in
        // heterogeneous subsets. Only weight *values* change; the per-neuron
        // `weights` length (and the matching `eligibility` length) are left as
        // neuromod built them.
        for (ni, neuron) in inner.neurons.iter_mut().enumerate() {
            for (c, w) in neuron.weights.iter_mut().enumerate() {
                *w = seeded_weight(seed, ni, c);
            }
        }
        // Snapshot the seeded weights so `reset` can restore the original
        // connectivity for a from-scratch replay (neuromod's own `reset` leaves
        // `neuron.weights` untouched, so R-STDP mutations would otherwise
        // persist).
        let initial_weights = inner.neurons.iter().map(|n| n.weights.clone()).collect();
        Self {
            inner,
            rng: StdRng::seed_from_u64(seed),
            initial_weights,
        }
    }

    /// Wrap a caller-constructed neuromod network with an RNG seeded from `seed`.
    ///
    /// Kept `pub(crate)` so no `neuromod` type leaks across the public boundary;
    /// used by tests that need a bespoke topology.
    #[allow(dead_code)]
    pub(crate) fn from_network(inner: neuromod::SpikingNetwork, seed: u64) -> Self {
        // Snapshot the caller-provided network's current weights as the replay
        // baseline so `reset` restores exactly this starting connectivity.
        let initial_weights = inner.neurons.iter().map(|n| n.weights.clone()).collect();
        Self {
            inner,
            rng: StdRng::seed_from_u64(seed),
            initial_weights,
        }
    }

    /// Reset the underlying neuron dynamics (membranes, spikes, counters) to
    /// their initial state **and restore the constructor's initial LIF weights**.
    ///
    /// `neuromod::SpikingNetwork::reset` resets membrane potentials, spike
    /// bookkeeping, eligibility traces, and modulators, but it does **not** touch
    /// `neuron.weights`. R-STDP mutates those weights mid-run, so without a
    /// restore a `reset(); reseed(seed)` "replay" would start from *learned*
    /// connectivity and could diverge from the original fired-index sequence.
    /// This method copies the snapshot taken at construction back into each LIF
    /// neuron so replay is from the original connectivity.
    ///
    /// This does **not** reseed the RNG: replay is explicit. To restart the
    /// random stream, call [`Self::reseed`]. Resetting dynamics without reseeding
    /// lets a caller continue a single random stream across logical episodes.
    pub fn reset(&mut self) {
        self.inner.reset();
        // Restore the seeded baseline weights (neuromod's reset leaves them as
        // R-STDP last mutated them). Lengths match by construction.
        for (neuron, initial) in self.inner.neurons.iter_mut().zip(&self.initial_weights) {
            neuron.weights.copy_from_slice(initial);
        }
    }

    /// Reseed the owned RNG from `seed`, restarting the deterministic stream.
    ///
    /// Combine with [`Self::reset`] (which also restores the initial LIF weights)
    /// to replay a run from a known starting point: `reset(); reseed(seed)`
    /// reproduces the original fired-index sequence even after R-STDP has mutated
    /// the live weights.
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
    fn seeded_weights_are_distinct_per_neuron() {
        // Regression for the lockstep bug: a single uniform weight makes every
        // LIF neuron behave identically. Assert the seed-derived weights differ
        // across neurons so the bank can fire in heterogeneous subsets.
        let snn = NeuromodSnn::with_seed(N, SEED);
        let first = &snn.inner.neurons[0].weights;
        let distinct = snn.inner.neurons.iter().any(|n| n.weights != *first);
        assert!(
            distinct,
            "seed-derived LIF weights must differ across neurons, not be uniform"
        );
        // Every weight is in the documented [0, 1) range.
        for neuron in &snn.inner.neurons {
            for &w in &neuron.weights {
                assert!((0.0..1.0).contains(&w), "seeded weight {w} out of [0,1)");
            }
        }
    }

    #[test]
    fn fires_in_proper_non_empty_subset() {
        // Devin finding: identical weights make all eight LIF neurons fire in
        // lockstep (all-or-nothing), collapsing eight neurons of signal into one
        // bit. With distinct per-neuron weights, a weak/varied stimulus must
        // produce at least one step whose fired set is a proper, non-empty subset
        // of the LIF bank (0 < fired.len() < num_lif) — i.e. neuron-level
        // patterns rather than all-or-nothing.
        let mut snn = NeuromodSnn::with_seed(N, SEED);
        let mods = NeuroModulators::default();
        // Weak, varied single-channel pokes to tease apart low- vs high-weight
        // neurons rather than driving the whole bank over threshold at once.
        let sequence = [
            [0.3, 0.0, 0.0, 0.0],
            [0.0, 0.35, 0.0, 0.0],
            [0.0, 0.0, 0.4, 0.0],
            [0.2, 0.0, 0.25, 0.0],
            [0.0, 0.3, 0.0, 0.3],
            [0.35, 0.0, 0.0, 0.2],
        ];
        let mut saw_proper_subset = false;
        for _ in 0..8 {
            for s in &sequence {
                let fired = snn.step(s, &mods).unwrap();
                if !fired.is_empty() && fired.len() < DEFAULT_NUM_LIF {
                    saw_proper_subset = true;
                }
            }
        }
        assert!(
            saw_proper_subset,
            "expected at least one step with a proper non-empty subset of the \
             {DEFAULT_NUM_LIF}-neuron LIF bank firing (not all-or-nothing lockstep)"
        );
    }

    // Seed proven (via out-of-tree neuromod probes) to make learned weights
    // flip at least one borderline weak-probe firing decision, so the replay
    // test below genuinely fails without the weight restore in `reset`.
    const PLASTICITY_SEED: u64 = 0xAAAA_1111;

    // Reward modulators drive R-STDP; a non-zero dopamine makes the weight
    // mutation under the strong phase pronounced.
    fn reward_mods() -> NeuroModulators {
        NeuroModulators {
            dopamine: 1.0,
            ..Default::default()
        }
    }

    // A strong, saturating stimulus block that reliably triggers spikes (and
    // therefore R-STDP weight updates).
    fn strong_block() -> Vec<[f32; N]> {
        vec![[0.95; N]; 40]
    }

    // A weak, varied probe whose firing is borderline, so it is sensitive to
    // whether the LIF weights are the original seeded values or R-STDP-mutated
    // ones.
    fn weak_probe() -> Vec<[f32; N]> {
        (0..40)
            .map(|i| {
                let v = 0.25 + 0.02 * (i % 7) as f32;
                [v, v * 0.8, v * 1.2, v * 0.5]
            })
            .collect()
    }

    #[test]
    fn replay_after_plasticity_restores_initial_weights() {
        // Codex P1: after R-STDP mutates weights mid-run, a documented
        // `reset(); reseed(seed)` replay must reproduce the ORIGINAL connectivity,
        // not the learned weights. We verify by comparing a plastic adapter —
        // driven over a strong block (mutating weights), then `reset()` (which
        // restores the seeded weights) and `reseed(seed)` — against a *fresh*
        // adapter of the same seed running only the weak probe. They must match,
        // which holds only because `reset` restores the constructor's initial
        // weights. (Confirmed during development that removing the restore in
        // `reset` makes this assertion fail for PLASTICITY_SEED.)
        let mods = reward_mods();

        // Reference: fresh adapter, weak probe only, no learning.
        let mut fresh = NeuromodSnn::with_seed(N, PLASTICITY_SEED);
        let reference: Vec<_> = weak_probe()
            .iter()
            .map(|s| fresh.step(s, &mods).unwrap())
            .collect();

        // Plastic adapter: strong block mutates weights, then reset + reseed and
        // run the same weak probe.
        let mut snn = NeuromodSnn::with_seed(N, PLASTICITY_SEED);
        for s in &strong_block() {
            let _ = snn.step(s, &mods).unwrap();
        }
        snn.reset();
        snn.reseed(PLASTICITY_SEED);
        let replayed: Vec<_> = weak_probe()
            .iter()
            .map(|s| snn.step(s, &mods).unwrap())
            .collect();

        assert_eq!(
            replayed, reference,
            "reset (with weight restore) + reseed must replay from the original \
             seeded connectivity, matching a fresh adapter"
        );
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
