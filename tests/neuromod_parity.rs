// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(all(feature = "neuromod", feature = "backends"))]

//! Parity / smoke test for the `neuromod` adapter against the `SimpleSnn`
//! reference mock.
//!
//! This does **not** assert bit-level equality between the two engines: they
//! implement entirely different dynamics (neuromod is a real LIF/Izhikevich
//! engine, `SimpleSnn` is a deterministic mock). It asserts that both honor the
//! same [`SpikingNetwork`] contract — shapes, width validation, error behaviour —
//! and that the adapter replays deterministically under a fixed seed.

use hybrid_fusion::{HybridError, NeuroModulators, NeuromodSnn, SimpleSnn, SpikingNetwork};

const N: usize = 4;
const SEED: u64 = 0x5EED_1234;

fn stimulus_sequence() -> Vec<[f32; N]> {
    vec![
        [0.9, 0.1, 0.5, 0.3],
        [0.2, 0.8, 0.4, 0.6],
        [0.7, 0.7, 0.1, 0.9],
        [0.3, 0.5, 0.9, 0.2],
        [0.6, 0.4, 0.8, 0.1],
    ]
}

#[test]
fn both_backends_return_indices_within_range() {
    let mods = NeuroModulators::default();
    let mut adapter = NeuromodSnn::with_seed(N, SEED);
    let mut mock = SimpleSnn::new(N);

    assert_eq!(adapter.num_channels(), N);
    assert_eq!(mock.num_channels(), N);

    // The adapter's fired indices are LIF-bank indices. The bank is
    // `min(num_channels, MAX_LIF_NEURONS)`, so every fired index is
    // `< num_channels()` and the reported width stays the input width.
    // Bound against `num_channels()`, and separately confirm at least one spike
    // occurs.
    let mut adapter_fired = false;
    for s in stimulus_sequence() {
        let a = adapter.step(&s, &mods).unwrap();
        let m = mock.step(&s, &mods).unwrap();
        assert!(
            a.iter().all(|&i| i < adapter.num_channels()),
            "adapter indices < num_channels()"
        );
        assert!(m.iter().all(|&i| i < N), "mock indices in range");
        adapter_fired |= !a.is_empty();
    }
    // Regression: the neuromod adapter must actually spike under stimulus (it
    // silently never fired while its LIF weights were left at the all-zero
    // neuromod default).
    assert!(
        adapter_fired,
        "neuromod adapter must fire at least once over the stimulus sequence"
    );
}

#[test]
fn both_backends_reject_wrong_width() {
    let mods = NeuroModulators::default();
    let mut adapter = NeuromodSnn::with_seed(N, SEED);
    let mut mock = SimpleSnn::new(N);

    let bad = [0.1, 0.2]; // width 2 != N

    match adapter.step(&bad, &mods).unwrap_err() {
        HybridError::InputLengthMismatch { expected, got } => {
            assert_eq!(expected, N);
            assert_eq!(got, 2);
        }
        other => panic!("adapter: expected InputLengthMismatch, got {other:?}"),
    }

    match mock.step(&bad, &mods).unwrap_err() {
        HybridError::InputLengthMismatch { expected, got } => {
            assert_eq!(expected, N);
            assert_eq!(got, 2);
        }
        other => panic!("mock: expected InputLengthMismatch, got {other:?}"),
    }
}

#[test]
fn adapter_rejects_non_finite_stimulus() {
    let mods = NeuroModulators::default();
    let mut adapter = NeuromodSnn::with_seed(N, SEED);
    let mut stim = [0.5; N];
    stim[2] = f32::NAN;
    let err = adapter.step(&stim, &mods).unwrap_err();
    assert!(
        matches!(err, HybridError::SnnStep(_)),
        "NaN stimulus must yield an Err (SnnStep), got {err:?}"
    );
}

#[test]
fn adapter_replays_deterministically_under_fixed_seed() {
    let mods = NeuroModulators::default();
    let run = || {
        let mut adapter = NeuromodSnn::with_seed(N, SEED);
        stimulus_sequence()
            .iter()
            .map(|s| adapter.step(s, &mods).unwrap())
            .collect::<Vec<_>>()
    };
    assert_eq!(run(), run(), "same seed => identical fired-index sequence");
}
