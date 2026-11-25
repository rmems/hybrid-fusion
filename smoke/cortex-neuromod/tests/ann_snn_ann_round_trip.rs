// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cortex-only ANN→SNN→ANN round trip on the CPU Vec reference backend
//! (Linear RM-2208).
//!
//! The cortex side is published `cortex-hybrid` 0.2.0 from crates.io, verified
//! on 2026-11-25. The fixed fixture is:
//!
//! - ANN ingress: `ReferenceExecutor` embedding stage, token ids `[1, 3, 0]`,
//!   output shape `[3, 4]`, mean-pooled to a 4-channel ANN signal.
//! - ANN→SNN: cortex `snn::AxonEncoder` wrapping an axon-encoder 0.6
//!   `EmbeddingRateEncoder` with `v_th = 0.4`.
//! - SNN: cortex `snn::NeuromodNetwork` with 4 LIF neurons, 1 auxiliary
//!   Izhikevich neuron, 4 channels, seed `0x2208_5EED`, stepped for 5 frozen
//!   ticks (`step_frozen`, so no plasticity is retained).
//! - SNN→ANN: cortex `snn::SpikeCountDecoder`; the 5 tick readouts are summed
//!   into one `[4]` readout and reshaped to `[1, 4]`.
//! - ANN egress: `ReferenceExecutor` readout stage with a fixed `[4, 2]`
//!   projection, output shape `[1, 2]`.
//!
//! Expected spike traces, intermediate stimuli, readouts, and output values are
//! checked in under `tests/fixtures/ann_snn_ann_round_trip.json`.

use axon_encoder::prelude::{EmbeddingEncoderConfig, EmbeddingRateEncoder};
use cortex_hybrid::snn::{
    AxonEncoder as CortexAxonEncoder, NeuromodNetwork, SnnBackend, SnnDecoder, SnnEncoder,
    SpikeCountDecoder,
};
use cortex_hybrid::stage::{ReferenceStageParams, StageInput, StageSource};
use cortex_hybrid::{
    AnnExecutor, AnnStage, CortexError, ReferenceExecutor, StageId, StageKind, Tensor,
};
use serde_json::{Value, json};

const DIM: usize = 4;
const CHANNELS: usize = 4;
const OUT_DIM: usize = 2;
const TICKS: usize = 5;
const SEED: u64 = 0x2208_5EED;
const TOKEN_IDS: [u32; 3] = [1, 3, 0];

#[derive(Debug, Clone, PartialEq)]
struct RoundTrip {
    hidden_shape: Vec<usize>,
    signal: Vec<f32>,
    stimuli: Vec<Vec<f32>>,
    spikes: Vec<Vec<usize>>,
    readout: Vec<f32>,
    output_shape: Vec<usize>,
    output: Vec<f32>,
}

#[derive(Debug, PartialEq)]
struct BackendState {
    global_step: i64,
    input_spike_times: Vec<i64>,
    predictive_state: Vec<f32>,
    membrane_potentials: Vec<f32>,
    weights: Vec<Vec<f32>>,
    thresholds: Vec<f32>,
    last_spikes: Vec<bool>,
}

fn token_table() -> Tensor {
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

fn embedding_hidden() -> Tensor {
    let tok = token_table();
    let pos = position_table();
    let id = StageId::new("embedding").expect("stage id");
    let stage = AnnStage {
        id: id.clone(),
        kind: StageKind::Embedding,
        inputs: vec![StageSource::External("tokens".into())],
    };
    let mut executor = ReferenceExecutor::new().bind(
        id,
        ReferenceStageParams::Embedding {
            tok_embed: &tok,
            pos_embed: &pos,
            max_seq_len: 4,
        },
    );
    executor
        .execute(&stage, &[StageInput::Tokens(&TOKEN_IDS)])
        .expect("embedding stage")
}

fn mean_pool_hidden(hidden: &Tensor) -> Vec<f32> {
    assert_eq!(hidden.shape(), &[TOKEN_IDS.len(), DIM]);
    let rows = hidden.shape()[0];
    let dim = hidden.shape()[1];
    (0..dim)
        .map(|col| {
            (0..rows)
                .map(|row| hidden.data()[row * dim + col])
                .sum::<f32>()
                / rows as f32
        })
        .collect()
}

fn readout_weight() -> Tensor {
    Tensor::try_from_vec(
        vec![
            0.25, -0.50, //
            0.50, 0.75, //
            -0.25, 1.00, //
            0.80, -0.40,
        ],
        &[CHANNELS, OUT_DIM],
    )
    .expect("readout weight")
}

fn bind_readout(lm_head: &Tensor) -> (ReferenceExecutor<'_>, AnnStage) {
    let id = StageId::new("snn_readout").expect("stage id");
    let stage = AnnStage {
        id: id.clone(),
        kind: StageKind::Readout,
        inputs: vec![StageSource::External("decoded_snn".into())],
    };
    let executor = ReferenceExecutor::new().bind(id, ReferenceStageParams::Readout { lm_head });
    (executor, stage)
}

fn make_encoder() -> CortexAxonEncoder<EmbeddingRateEncoder> {
    CortexAxonEncoder::new(
        EmbeddingRateEncoder::try_new(CHANNELS, EmbeddingEncoderConfig { v_th: 0.4 })
            .expect("embedding rate encoder"),
    )
}

fn make_backend() -> NeuromodNetwork {
    NeuromodNetwork::new(CHANNELS, 1, CHANNELS)
        .expect("neuromod network")
        .with_seed(SEED)
}

fn run_round_trip(
    encoder: &mut CortexAxonEncoder<EmbeddingRateEncoder>,
    backend: &mut NeuromodNetwork,
) -> RoundTrip {
    let hidden = embedding_hidden();
    let signal = mean_pool_hidden(&hidden);
    assert_eq!(signal.len(), CHANNELS);
    assert!(signal.iter().all(|v| v.is_finite()));

    let mut stimuli = Vec::with_capacity(TICKS);
    let mut spikes = Vec::with_capacity(TICKS);
    let mut readout = vec![0.0f32; CHANNELS];
    for _tick in 0..TICKS {
        let stimulus = encoder.encode(&signal, CHANNELS).expect("encode one tick");
        assert_eq!(stimulus.len(), CHANNELS);
        assert!(stimulus.iter().all(|v| v.is_finite()));
        let step = backend
            .step_frozen(&stimulus)
            .expect("frozen neuromod tick");
        assert!(
            step.spikes.iter().all(|&idx| idx < CHANNELS),
            "spike outside channel space: {:?}",
            step.spikes
        );
        let decoded = SpikeCountDecoder
            .decode(&step, CHANNELS)
            .expect("decode one tick");
        assert_eq!(decoded.len(), CHANNELS);
        for (acc, value) in readout.iter_mut().zip(decoded) {
            *acc += value;
        }
        stimuli.push(stimulus);
        spikes.push(step.spikes);
    }

    let ann_input = Tensor::try_from_vec(readout.clone(), &[1, CHANNELS]).expect("readout tensor");
    let lm_head = readout_weight();
    let (mut executor, stage) = bind_readout(&lm_head);
    let output = executor
        .execute(&stage, &[StageInput::Tensor(&ann_input)])
        .expect("readout stage");

    RoundTrip {
        hidden_shape: hidden.shape().to_vec(),
        signal,
        stimuli,
        spikes,
        readout,
        output_shape: output.shape().to_vec(),
        output: output.data().to_vec(),
    }
}

fn usize_field(value: &Value, key: &str) -> usize {
    value[key]
        .as_u64()
        .unwrap_or_else(|| panic!("fixture {key} must be an unsigned integer")) as usize
}

fn usize_vec(value: &Value, key: &str) -> Vec<usize> {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("fixture {key} must be an array"))
        .iter()
        .map(|item| {
            item.as_u64()
                .unwrap_or_else(|| panic!("fixture {key} entries must be unsigned integers"))
                as usize
        })
        .collect()
}

fn f32_vec(value: &Value, key: &str) -> Vec<f32> {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("fixture {key} must be an array"))
        .iter()
        .map(|item| {
            item.as_f64()
                .unwrap_or_else(|| panic!("fixture {key} entries must be numbers"))
                as f32
        })
        .collect()
}

fn nested_usize_vec(value: &Value, key: &str) -> Vec<Vec<usize>> {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("fixture {key} must be an array"))
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap_or_else(|| panic!("fixture {key} rows must be arrays"))
                .iter()
                .map(|item| {
                    item.as_u64().unwrap_or_else(|| {
                        panic!("fixture {key} entries must be unsigned integers")
                    }) as usize
                })
                .collect()
        })
        .collect()
}

fn nested_f32_vec(value: &Value, key: &str) -> Vec<Vec<f32>> {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("fixture {key} must be an array"))
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap_or_else(|| panic!("fixture {key} rows must be arrays"))
                .iter()
                .map(|item| {
                    item.as_f64()
                        .unwrap_or_else(|| panic!("fixture {key} entries must be numbers"))
                        as f32
                })
                .collect()
        })
        .collect()
}

fn assert_matches_golden(actual: &RoundTrip) {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/ann_snn_ann_round_trip.json"))
        .expect("parse round-trip fixture");
    let matches = [
        usize_vec(&fixture, "input_shape") == actual.hidden_shape,
        usize_field(&fixture, "channels") == CHANNELS,
        usize_field(&fixture, "lif_neurons") == CHANNELS,
        usize_field(&fixture, "izh_neurons") == 1,
        usize_field(&fixture, "ticks") == TICKS,
        fixture["seed"].as_u64() == Some(SEED),
        f32_vec(&fixture, "ann_signal") == actual.signal,
        nested_f32_vec(&fixture, "encoded_stimuli") == actual.stimuli,
        nested_usize_vec(&fixture, "spike_trace") == actual.spikes,
        f32_vec(&fixture, "snn_readout") == actual.readout,
        usize_vec(&fixture, "output_shape") == actual.output_shape,
        f32_vec(&fixture, "ann_output") == actual.output,
    ]
    .into_iter()
    .all(|matches| matches);

    if !matches {
        let actual_json = json!({
            "input_shape": actual.hidden_shape,
            "channels": CHANNELS,
            "lif_neurons": CHANNELS,
            "izh_neurons": 1,
            "ticks": TICKS,
            "seed": SEED,
            "ann_signal": actual.signal,
            "encoded_stimuli": actual.stimuli,
            "spike_trace": actual.spikes,
            "snn_readout": actual.readout,
            "output_shape": actual.output_shape,
            "ann_output": actual.output,
        });
        panic!(
            "round-trip golden mismatch; actual fixture is:\n{}",
            serde_json::to_string_pretty(&actual_json).expect("serialize actual fixture")
        );
    }
}

fn backend_state(backend: &NeuromodNetwork) -> BackendState {
    let network = backend.network();
    BackendState {
        global_step: network.global_step,
        input_spike_times: network.input_spike_times.clone(),
        predictive_state: network.predictive_state.clone(),
        membrane_potentials: network
            .neurons
            .iter()
            .map(|neuron| neuron.membrane_potential)
            .collect(),
        weights: network
            .neurons
            .iter()
            .map(|neuron| neuron.weights.clone())
            .collect(),
        thresholds: network
            .neurons
            .iter()
            .map(|neuron| neuron.threshold)
            .collect(),
        last_spikes: network
            .neurons
            .iter()
            .map(|neuron| neuron.last_spike)
            .collect(),
    }
}

#[test]
fn cortex_only_round_trip_matches_checked_in_golden() {
    let mut encoder = make_encoder();
    let mut backend = make_backend();
    let actual = run_round_trip(&mut encoder, &mut backend);
    assert_matches_golden(&actual);
}

#[test]
fn same_seed_and_reset_reproduce_the_round_trip() {
    let mut first_encoder = make_encoder();
    let mut first_backend = make_backend();
    let mut second_encoder = make_encoder();
    let mut second_backend = make_backend();

    let first = run_round_trip(&mut first_encoder, &mut first_backend);
    let second = run_round_trip(&mut second_encoder, &mut second_backend);
    assert_eq!(first, second, "fresh same-seed runs diverged");

    first_encoder.reset();
    first_backend.reset();
    let replay = run_round_trip(&mut first_encoder, &mut first_backend);
    assert_eq!(replay, first, "reset run differed from the fresh instance");
}

#[test]
fn mismatched_channel_count_is_a_typed_error_not_a_panic() {
    let mut backend = make_backend();
    let err = backend
        .step_frozen(&[0.0; CHANNELS - 1])
        .expect_err("short stimulus must be rejected");
    assert!(matches!(
        err,
        CortexError::InputLengthMismatch {
            expected: CHANNELS,
            got: 3
        }
    ));
}

#[test]
fn rejected_inputs_leave_executor_and_backend_state_unchanged() {
    let lm_head = readout_weight();
    let (mut executor, stage) = bind_readout(&lm_head);
    let valid =
        Tensor::try_from_vec(vec![1.0, 0.0, 2.0, 1.0], &[1, CHANNELS]).expect("valid input");
    let invalid = Tensor::try_from_vec(vec![1.0, 0.0, 2.0], &[1, CHANNELS - 1]).expect("bad width");

    let err = executor
        .execute(&stage, &[StageInput::Tensor(&invalid)])
        .expect_err("wrong readout width must be rejected");
    assert!(
        matches!(err, CortexError::StageFailed { .. }),
        "expected typed StageFailed, got {err:?}"
    );
    let after_reject = executor
        .execute(&stage, &[StageInput::Tensor(&valid)])
        .expect("executor remains usable");

    let (mut control_executor, control_stage) = bind_readout(&lm_head);
    let control = control_executor
        .execute(&control_stage, &[StageInput::Tensor(&valid)])
        .expect("control executor output");
    assert_eq!(after_reject.data(), control.data());

    // Fractional stimuli exercise the seeded Bernoulli input encoding. If the
    // rejected call consumed RNG draws, the post-rejection sequence diverges
    // from the control backend with high probability; without a regression the
    // same seed replays exactly.
    let stimulus = vec![0.5f32; CHANNELS];
    let mut backend = make_backend();
    let mut control_backend = make_backend();
    backend.step_frozen(&stimulus).expect("backend setup tick");
    control_backend
        .step_frozen(&stimulus)
        .expect("control setup tick");
    let before = backend_state(&backend);

    let err = backend
        .step_frozen(&stimulus[..CHANNELS - 1])
        .expect_err("short stimulus must be rejected");
    assert!(matches!(err, CortexError::InputLengthMismatch { .. }));
    assert_eq!(backend_state(&backend), before);

    for _ in 0..4 {
        let after = backend
            .step_frozen(&stimulus)
            .expect("backend remains usable");
        let control_after = control_backend
            .step_frozen(&stimulus)
            .expect("control backend output");
        assert_eq!(
            after.spikes, control_after.spikes,
            "rejected input consumed backend state or RNG"
        );
    }
    assert_eq!(
        backend_state(&backend),
        backend_state(&control_backend),
        "rejected input changed the post-recovery backend state"
    );
}
