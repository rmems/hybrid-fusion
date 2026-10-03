// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "cortex-smoke")]

//! Fixed-seed smoke test: one real cortex-tensor embedding stage into one real
//! `neuromod` 0.7 network through [`hybrid_fusion::NeuromodSnn`] (issue #45).
//!
//! Compile and run with `cargo test --features cortex-smoke`. That feature
//! implies `neuromod`. The default build and `cargo test --features neuromod`
//! do not compile this file and do not fetch cortex-tensor.
//!
//! # Fixture
//!
//! - ANN: cortex-tensor `ReferenceExecutor` embedding stage. Token table and
//!   position table are fixed literals (vocab 4, dim 4, max sequence 4). Input
//!   token ids are `[1, 3, 0]`. Output is token embedding plus position
//!   embedding, shape `[3, 4]`.
//! - Boundary: [`embed_to_stimuli_with_width`] mean-pools that hidden state,
//!   resizes to 4 stimulus channels, and applies `tanh`.
//! - SNN: [`NeuromodSnn::with_seed`] with 4 input channels, 4 output neurons,
//!   seed `0x45_5EE5`. The same projected stimulus is applied for 4 steps.
//!   `step` goes through the adapter's owned RNG (`step_with_rng`).
//!
//! # Spike envelope
//!
//! Each step can fire at most 4 neurons, so 4 steps can fire at most 16 times.
//! The total must be in `1..=16`. All-zero is a failure: the tables and weights
//! are chosen so a positive stimulus reaches a network whose synapses are
//! strictly positive. The bound is loose on purpose — this is not a spike-count
//! benchmark.

use cortex_tensor::stage::{ReferenceStageParams, StageInput, StageSource};
use cortex_tensor::{AnnExecutor, AnnStage, ReferenceExecutor, StageId, StageKind, Tensor};
use hybrid_fusion::projector::embed_to_stimuli_with_width;
use hybrid_fusion::{NeuroModulators, NeuromodSnn, SpikingNetwork, Tensor as HybridTensor};

const DIM: usize = 4;
const CHANNELS: usize = 4;
const STEPS: usize = 4;
const SEED: u64 = 0x45_5EE5;
const OTHER_SEED: u64 = 0x45_0EED;
/// Loose spike-count envelope: not silent, and not more fires than neurons × steps.
const SPIKE_TOTAL_MIN: usize = 1;
const SPIKE_TOTAL_MAX: usize = CHANNELS * STEPS;

const TOKEN_IDS: [u32; 3] = [1, 3, 0];

fn token_table() -> Tensor {
    // [vocab=4, dim=4], hand-written, all finite and positive.
    Tensor::try_from_vec(
        vec![
            0.10, 0.20, 0.30, 0.40, //
            0.50, 0.15, 0.25, 0.35, //
            0.05, 0.45, 0.20, 0.10, //
            0.30, 0.30, 0.55, 0.20,
        ],
        &[4, DIM],
    )
    .expect("token table")
}

fn position_table() -> Tensor {
    // [max_seq=4, dim=4]
    Tensor::try_from_vec(
        vec![
            0.01, 0.02, 0.00, 0.03, //
            0.04, 0.00, 0.02, 0.01, //
            0.00, 0.03, 0.01, 0.02, //
            0.02, 0.01, 0.00, 0.04,
        ],
        &[4, DIM],
    )
    .expect("position table")
}

fn hidden_state(tok: &Tensor, pos: &Tensor) -> Tensor {
    let id = StageId::new("embedding").expect("stage id");
    let stage = AnnStage {
        id: id.clone(),
        kind: StageKind::Embedding,
        inputs: vec![StageSource::External("tokens".into())],
    };
    let mut executor = ReferenceExecutor::new().bind(
        id,
        ReferenceStageParams::Embedding {
            tok_embed: tok,
            pos_embed: pos,
            max_seq_len: 4,
        },
    );
    executor
        .execute(&stage, &[StageInput::Tokens(&TOKEN_IDS)])
        .expect("embedding stage")
}

fn project(hidden: &Tensor) -> Vec<f32> {
    let hybrid = HybridTensor::from_vec(hidden.data().to_vec(), hidden.shape());
    embed_to_stimuli_with_width(&hybrid, CHANNELS)
}

fn run(snn: &mut NeuromodSnn, stimulus: &[f32]) -> Vec<Vec<usize>> {
    let mods = NeuroModulators::default();
    (0..STEPS)
        .map(|_| snn.step(stimulus, &mods).expect("snn step"))
        .collect()
}

fn assert_finite(values: &[f32], what: &str) {
    for (i, v) in values.iter().enumerate() {
        assert!(v.is_finite(), "{what}[{i}] = {v} is not finite");
    }
}

fn assert_in_range(fired: &[Vec<usize>]) {
    for (step, indices) in fired.iter().enumerate() {
        for &idx in indices {
            assert!(
                idx < CHANNELS,
                "step {step} fired index {idx} is outside 0..{CHANNELS}"
            );
        }
    }
    let total: usize = fired.iter().map(Vec::len).sum();
    assert!(
        (SPIKE_TOTAL_MIN..=SPIKE_TOTAL_MAX).contains(&total),
        "spike total {total} outside documented envelope {SPIKE_TOTAL_MIN}..={SPIKE_TOTAL_MAX}"
    );
}

#[test]
fn cortex_embedding_projects_to_finite_stimuli_and_in_range_spikes() {
    let tok = token_table();
    let pos = position_table();
    let hidden = hidden_state(&tok, &pos);
    assert_eq!(hidden.shape(), &[TOKEN_IDS.len(), DIM]);
    assert_finite(hidden.data(), "hidden");

    let stimulus = project(&hidden);
    assert_eq!(stimulus.len(), CHANNELS);
    assert_finite(&stimulus, "stimulus");
    // tanh of a positive pool stays in (0, 1). All-zero stimuli would make an
    // all-zero spike count unsurprising; this fixture must not be that case.
    assert!(
        stimulus.iter().any(|v| *v > 0.0),
        "projected stimulus is entirely non-positive: {stimulus:?}"
    );

    let mut snn = NeuromodSnn::with_seed(CHANNELS, CHANNELS, SEED).expect("network");
    assert_eq!(snn.num_channels(), CHANNELS);
    assert_eq!(snn.num_neurons(), CHANNELS);
    let fired = run(&mut snn, &stimulus);
    assert_in_range(&fired);
}

#[test]
fn same_seed_reproduces_fired_indices_including_after_reset() {
    let stimulus = project(&hidden_state(&token_table(), &position_table()));

    let mut first = NeuromodSnn::with_seed(CHANNELS, CHANNELS, SEED).unwrap();
    let mut second = NeuromodSnn::with_seed(CHANNELS, CHANNELS, SEED).unwrap();
    let a = run(&mut first, &stimulus);
    let b = run(&mut second, &stimulus);
    assert_eq!(a, b, "two networks with the same seed diverged");
    assert_eq!(
        a.iter().map(Vec::len).collect::<Vec<_>>(),
        b.iter().map(Vec::len).collect::<Vec<_>>()
    );

    // reset() restores dynamics and the construction-time weights. It does not
    // rewind the owned RNG; re-running with the same seed means reseed(SEED).
    first.reset();
    first.reseed(SEED);
    let replay = run(&mut first, &stimulus);
    assert_eq!(
        replay, a,
        "reset + same seed did not reproduce the first run"
    );
}

#[test]
fn a_different_seed_is_not_required_to_match() {
    let stimulus = project(&hidden_state(&token_table(), &position_table()));
    let mut seeded = NeuromodSnn::with_seed(CHANNELS, CHANNELS, SEED).unwrap();
    let mut other = NeuromodSnn::with_seed(CHANNELS, CHANNELS, OTHER_SEED).unwrap();
    let a = run(&mut seeded, &stimulus);
    let b = run(&mut other, &stimulus);
    assert_in_range(&a);
    assert_in_range(&b);
    // Issue #45: a different seed is allowed to differ. It is not required to.
    // If these two seeds happen to emit the same indices, that is still a pass;
    // asserting inequality would freeze an accident of the weight hash.
    let _allowed_to_differ = a != b;
}
