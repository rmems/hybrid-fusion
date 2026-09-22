// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration tests for `HybridNetwork` using mock backends.
//!
//! These tests exercise only the public API surface — no `pub(crate)` or internal access —
//! and serve as usage examples for consumers of the crate.
//!
//! Note: `HybridNetwork::{transformer, snn}` are public fields and are part of the public
//! surface; tests prefer `config()` / output fields when possible.

use hybrid_fusion::{
    ForwardValueStage, HybridConfig, HybridError, HybridNetwork, HybridOutput, NeuroModulators,
    SpikingNetwork, Tensor, Transformer,
};
use std::cell::Cell;

// ---------------------------------------------------------------------------
// Mock backends
// ---------------------------------------------------------------------------

struct MockTransformer {
    dim: usize,
    max_seq: usize,
}

impl Transformer for MockTransformer {
    fn hidden_states(&self, token_ids: &[u32]) -> Tensor {
        let seq = token_ids.len();
        // Guard the Tensor >0 dim invariant even though HybridNetwork::forward
        // rejects empty token_ids before calling this method.
        assert!(seq > 0, "token_ids must not be empty");
        assert!(self.dim > 0, "transformer dim must be > 0");
        let data: Vec<f32> = (0..seq * self.dim).map(|i| (i as f32) * 0.01).collect();
        Tensor::from_vec(data, &[seq, self.dim])
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq
    }

    fn param_count(&self) -> usize {
        self.dim * 1000
    }
}

struct MockSnn {
    channels: usize,
    /// Last modulators observed by `step` (for custom-modulator propagation).
    last_modulators: Option<NeuroModulators>,
}

impl MockSnn {
    fn new(channels: usize) -> Self {
        Self {
            channels,
            last_modulators: None,
        }
    }
}

impl SpikingNetwork for MockSnn {
    fn step(
        &mut self,
        stimuli: &[f32],
        modulators: &NeuroModulators,
    ) -> hybrid_fusion::Result<Vec<usize>> {
        // Record received modulators so integration tests can verify forward
        // actually propagates custom values (not just defaults).
        self.last_modulators = Some(modulators.clone());
        Ok(stimuli
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > 0.0)
            .map(|(i, _)| i)
            .collect())
    }

    fn num_channels(&self) -> usize {
        self.channels
    }
}

/// Transformer that reports a positive dim but emits a mismatched hidden width
/// so `HybridNetwork::forward` hits the InvalidConfig path.
struct MismatchedDimTransformer {
    reported_dim: usize,
    actual_dim: usize,
    max_seq: usize,
}

impl Transformer for MismatchedDimTransformer {
    fn hidden_states(&self, token_ids: &[u32]) -> Tensor {
        let seq = token_ids.len().max(1);
        let data = vec![0.1f32; seq * self.actual_dim];
        Tensor::from_vec(data, &[seq, self.actual_dim])
    }

    fn dim(&self) -> usize {
        self.reported_dim
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq
    }

    fn param_count(&self) -> usize {
        self.reported_dim
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a network whose mock SNN channel count is deliberately *different*
/// from `cfg.snn_input_channels`, so shape assertions exercise
/// `SpikingNetwork::num_channels()` wiring rather than config defaults.
fn build_network() -> HybridNetwork<MockTransformer, MockSnn> {
    let cfg = HybridConfig::tiny();
    // Leave cfg.snn_input_channels at its default; mock uses a different width.
    let mock_channels = cfg.snn_input_channels.saturating_add(13).max(1);
    assert_ne!(mock_channels, cfg.snn_input_channels);
    let t = MockTransformer {
        dim: cfg.transformer.dim,
        max_seq: cfg.transformer.max_seq_len,
    };
    let s = MockSnn::new(mock_channels);
    HybridNetwork::new(t, s, cfg)
}

fn modulators_eq(a: &NeuroModulators, b: &NeuroModulators) -> bool {
    a.dopamine == b.dopamine
        && a.cortisol == b.cortisol
        && a.acetylcholine == b.acetylcholine
        && a.tempo == b.tempo
        && a.aux_dopamine == b.aux_dopamine
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_forward_output_shapes() {
    let mut net = build_network();
    let snn_channels = net.snn.num_channels();
    // Prove the mock width is not the config default (decoupled wiring).
    assert_ne!(snn_channels, net.config().snn_input_channels);

    let out: HybridOutput = net
        .forward(&[1, 2, 3, 4], None)
        .expect("forward should succeed");

    // embedding dimension matches the transformer's hidden dim (trait contract)
    assert_eq!(out.embedding.len(), net.transformer.dim());

    // stimuli dimension matches the SNN's public channel count, not config
    assert_eq!(out.stimuli.len(), snn_channels);
    assert_ne!(out.stimuli.len(), net.config().snn_input_channels);

    // fired_neurons is a subset of valid channel indices
    for &idx in &out.fired_neurons {
        assert!(idx < snn_channels, "fired neuron index out of range");
    }

    // first forward sets global_step to 1
    assert_eq!(out.global_step, 1);

    // ANN→SNN path does not populate reverse-path MoE fields
    assert!(out.expert_weights.is_none());
    assert!(out.selected_experts.is_none());
    assert!(out.routing_entropy.is_none());
}

#[test]
fn test_spike_activity_from_fired() {
    use hybrid_fusion::SpikeActivity;
    let act = SpikeActivity::from_fired(&[1, 3], 8).expect("valid from_fired");
    assert_eq!(act.spike_train, vec![vec![1, 3]]);
    assert_eq!(act.potentials.len(), 8);
    assert!(act.iz_potentials.is_empty());
}

#[test]
fn test_spike_activity_from_fired_edges() {
    use hybrid_fusion::{HybridError, SpikeActivity};

    // empty fired + valid neuron count
    let silent = SpikeActivity::from_fired(&[], 8).unwrap();
    assert_eq!(silent.spike_train, vec![Vec::<usize>::new()]);
    assert_eq!(silent.potentials.len(), 8);

    // zero neurons rejected
    match SpikeActivity::from_fired(&[], 0).unwrap_err() {
        HybridError::InvalidConfig(_) => {}
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
    match SpikeActivity::from_fired(&[0], 0).unwrap_err() {
        HybridError::InvalidConfig(_) => {}
        other => panic!("expected InvalidConfig, got {other:?}"),
    }

    // out-of-range index rejected
    match SpikeActivity::from_fired(&[8], 8).unwrap_err() {
        HybridError::InvalidConfig(_) => {}
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

#[test]
fn test_forward_rejects_empty() {
    let mut net = build_network();

    let err = net
        .forward(&[], None)
        .expect_err("empty token_ids should fail");
    match err {
        HybridError::InputLengthMismatch { expected, got } => {
            assert_eq!(expected, 1);
            assert_eq!(got, 0);
        }
        other => panic!("expected InputLengthMismatch, got {other:?}"),
    }
}

#[test]
fn test_forward_rejects_over_long() {
    let cfg = HybridConfig::tiny();
    let mut net = build_network();
    let too_long = vec![0u32; cfg.transformer.max_seq_len + 1];

    let err = net
        .forward(&too_long, None)
        .expect_err("over-length token_ids should fail");
    match err {
        HybridError::InputLengthMismatch { expected, got } => {
            assert_eq!(expected, cfg.transformer.max_seq_len);
            assert_eq!(got, too_long.len());
        }
        other => panic!("expected InputLengthMismatch, got {other:?}"),
    }
}

#[test]
fn test_global_step_increments_and_resets() {
    let mut net = build_network();

    assert_eq!(net.global_step(), 0, "initial step should be 0");

    net.forward(&[0, 1], None).unwrap();
    assert_eq!(net.global_step(), 1);

    net.forward(&[0, 1], None).unwrap();
    assert_eq!(net.global_step(), 2, "step should increment each forward");

    net.reset();
    assert_eq!(net.global_step(), 0, "reset should bring step back to 0");
}

#[test]
fn test_stimuli_bounded() {
    let mut net = build_network();

    // Use a variety of token counts to exercise different pooling paths
    for tokens in [1u32, 3, 8, 32] {
        let token_ids: Vec<u32> = (0..tokens).collect();
        let out = net.forward(&token_ids, None).unwrap();

        for (i, v) in out.stimuli.iter().enumerate() {
            assert!(
                v.abs() <= 1.0,
                "stimuli[{i}] = {v} is outside [-1, 1] for {tokens} tokens"
            );
        }
    }
}

#[test]
fn test_forward_with_custom_modulators() {
    let mut net = build_network();

    let custom = NeuroModulators {
        dopamine: 0.8,
        cortisol: 0.2,
        acetylcholine: 0.6,
        tempo: 1.5,
        aux_dopamine: 0.1,
    };

    let out = net
        .forward(&[10, 20, 30], Some(custom.clone()))
        .expect("forward with custom modulators should succeed");

    // Prefer trait contracts / mock observations over config defaults.
    assert_eq!(out.embedding.len(), net.transformer.dim());
    assert_eq!(out.stimuli.len(), net.snn.num_channels());
    assert_eq!(out.global_step, 1);

    // Mock must have observed the exact custom modulators (proves forward
    // propagates Some(custom) rather than always substituting defaults).
    let observed = net
        .snn
        .last_modulators
        .as_ref()
        .expect("MockSnn should have recorded modulators");
    assert!(
        modulators_eq(observed, &custom),
        "expected custom modulators to be forwarded, got {observed:?}"
    );
    assert!(
        !modulators_eq(observed, &NeuroModulators::default()),
        "custom modulators must not equal defaults for this test to be meaningful"
    );

    // stimuli must still be bounded even with non-default modulators
    for v in &out.stimuli {
        assert!(v.abs() <= 1.0);
    }
}

/// Zero-*extent* dimensions (any axis of length 0) are always an error.
/// Rank-0 tensors (`shape = &[]`) are a separate case: product is 1 and they
/// represent a scalar, which the public Tensor API accepts (see property tests).
#[test]
fn test_tensor_rejects_zero_extent_dimensions() {
    let cases: &[&[usize]] = &[&[0], &[1, 0], &[0, 4], &[2, 0, 3]];
    for shape in cases {
        let result = std::panic::catch_unwind(|| {
            let len: usize = shape.iter().product();
            Tensor::from_vec(vec![0.0; len], shape);
        });
        assert!(
            result.is_err(),
            "Tensor::from_vec should reject zero-extent shape {shape:?}"
        );

        let result = std::panic::catch_unwind(|| {
            Tensor::zeros(shape);
        });
        assert!(
            result.is_err(),
            "Tensor::zeros should reject zero-extent shape {shape:?}"
        );
    }
}

/// Rank-0 (`shape = &[]`) is accepted as a scalar tensor (numel == 1).
#[test]
fn test_tensor_accepts_rank0_scalar() {
    let t = Tensor::from_vec(vec![3.5], &[]);
    assert_eq!(t.ndim(), 0);
    assert_eq!(t.numel(), 1);
    assert_eq!(t.data(), &[3.5]);

    let z = Tensor::zeros(&[]);
    assert_eq!(z.ndim(), 0);
    assert_eq!(z.numel(), 1);
    assert_eq!(z.data(), &[0.0]);
}

/// Hidden-state width that disagrees with Transformer::dim() is rejected via
/// the public forward API rather than silently projected.
#[test]
fn test_forward_rejects_hidden_dim_mismatch() {
    let cfg = HybridConfig::tiny();
    let reported_dim = cfg.transformer.dim;
    let actual_dim = reported_dim + 8;
    let t = MismatchedDimTransformer {
        reported_dim,
        actual_dim,
        max_seq: cfg.transformer.max_seq_len,
    };
    let s = MockSnn::new(cfg.snn_input_channels);
    let mut net = HybridNetwork::new(t, s, cfg);

    let err = net
        .forward(&[1, 2, 3], None)
        .expect_err("mismatched hidden dim should fail");
    match err {
        HybridError::HiddenStateDim { expected, got } => {
            assert_eq!(expected, reported_dim);
            assert_eq!(got, actual_dim);
        }
        other => panic!("expected HiddenStateDim, got {other:?}"),
    }
    assert_eq!(net.global_step(), 0);
}

// ---------------------------------------------------------------------------
// Preflight contract errors (public API)
// ---------------------------------------------------------------------------

struct ScriptedTransformer {
    dim: usize,
    max_seq: usize,
    hidden: Tensor,
}

impl Transformer for ScriptedTransformer {
    fn hidden_states(&self, _token_ids: &[u32]) -> Tensor {
        self.hidden.clone()
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq
    }
    fn param_count(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// HybridNetwork::try_new construction validation
// ---------------------------------------------------------------------------

struct CountingTransformer {
    dim: usize,
    max_seq: usize,
    hidden_calls: Cell<usize>,
    param_calls: Cell<usize>,
}

impl Transformer for CountingTransformer {
    fn hidden_states(&self, token_ids: &[u32]) -> Tensor {
        self.hidden_calls.set(self.hidden_calls.get() + 1);
        let seq = token_ids.len().max(1);
        let data = vec![0.1f32; seq * self.dim.max(1)];
        Tensor::from_vec(data, &[seq, self.dim.max(1)])
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq
    }

    fn param_count(&self) -> usize {
        self.param_calls.set(self.param_calls.get() + 1);
        self.dim
    }
}

struct CountingSnn {
    channels: usize,
    step_calls: Cell<usize>,
}

impl SpikingNetwork for CountingSnn {
    fn step(
        &mut self,
        _stimuli: &[f32],
        _modulators: &NeuroModulators,
    ) -> hybrid_fusion::Result<Vec<usize>> {
        self.step_calls.set(self.step_calls.get() + 1);
        Ok(Vec::new())
    }

    fn num_channels(&self) -> usize {
        self.channels
    }
}

struct SpySnn {
    channels: usize,
    step_calls: usize,
}

impl SpySnn {
    fn new(channels: usize) -> Self {
        Self {
            channels,
            step_calls: 0,
        }
    }
}

impl SpikingNetwork for SpySnn {
    fn step(
        &mut self,
        _stimuli: &[f32],
        _modulators: &NeuroModulators,
    ) -> hybrid_fusion::Result<Vec<usize>> {
        self.step_calls += 1;
        Ok(Vec::new())
    }

    fn num_channels(&self) -> usize {
        self.channels
    }
}

fn construction_error<T, S>(transformer: T, snn: S, config: HybridConfig) -> HybridError
where
    T: Transformer,
    S: SpikingNetwork,
{
    match HybridNetwork::try_new(transformer, snn, config) {
        Ok(_) => panic!("expected HybridNetwork::try_new to fail"),
        Err(err) => err,
    }
}

fn matching_tiny() -> (HybridConfig, MockTransformer, MockSnn) {
    let cfg = HybridConfig::tiny();
    let t = MockTransformer {
        dim: cfg.transformer.dim,
        max_seq: cfg.transformer.max_seq_len,
    };
    let s = MockSnn::new(cfg.snn_input_channels);
    (cfg, t, s)
}

fn assert_config_mismatch(
    err: HybridError,
    field: &'static str,
    configured: usize,
    backend: usize,
) {
    match &err {
        HybridError::ConfigMismatch {
            field: got_field,
            configured: got_configured,
            backend: got_backend,
        } => {
            assert_eq!(*got_field, field, "mismatch field");
            assert_eq!(*got_configured, configured, "configured value");
            assert_eq!(*got_backend, backend, "backend value");
            assert_eq!(
                err.to_string(),
                format!(
                    "configuration mismatch for {field}: configured {configured}, backend {backend}"
                )
            );
        }
        other => panic!("expected ConfigMismatch, got {other:?}"),
    }
}

#[test]
fn test_try_new_rejects_transformer_dim_mismatch() {
    let (mut cfg, t, s) = matching_tiny();
    let backend = t.dim;
    cfg.transformer.dim = backend + 8;
    let err = construction_error(t, s, cfg);
    assert_config_mismatch(
        err,
        HybridError::FIELD_TRANSFORMER_DIM,
        backend + 8,
        backend,
    );
}

#[test]
fn test_try_new_rejects_max_seq_len_mismatch() {
    let (mut cfg, t, s) = matching_tiny();
    let backend = t.max_seq;
    cfg.transformer.max_seq_len = backend + 4;
    let err = construction_error(t, s, cfg);
    assert_config_mismatch(
        err,
        HybridError::FIELD_TRANSFORMER_MAX_SEQ_LEN,
        backend + 4,
        backend,
    );
}

#[test]
fn test_try_new_rejects_snn_channels_mismatch() {
    let (mut cfg, t, s) = matching_tiny();
    let backend = s.channels;
    cfg.snn_input_channels = backend + 13;
    let err = construction_error(t, s, cfg);
    assert_config_mismatch(
        err,
        HybridError::FIELD_SNN_INPUT_CHANNELS,
        backend + 13,
        backend,
    );
}

#[test]
fn test_try_new_rejects_zero_transformer_dim() {
    let (mut cfg, t, s) = matching_tiny();
    let backend = t.dim;
    cfg.transformer.dim = 0;
    let err = construction_error(t, s, cfg);
    assert_config_mismatch(err, HybridError::FIELD_TRANSFORMER_DIM, 0, backend);
}

#[test]
fn test_try_new_rejects_zero_max_seq_len() {
    let (mut cfg, t, s) = matching_tiny();
    let backend = t.max_seq;
    cfg.transformer.max_seq_len = 0;
    let err = construction_error(t, s, cfg);
    assert_config_mismatch(err, HybridError::FIELD_TRANSFORMER_MAX_SEQ_LEN, 0, backend);
}

#[test]
fn test_try_new_rejects_zero_snn_channels() {
    let (mut cfg, t, s) = matching_tiny();
    let backend = s.channels;
    cfg.snn_input_channels = 0;
    let err = construction_error(t, s, cfg);
    assert_config_mismatch(err, HybridError::FIELD_SNN_INPUT_CHANNELS, 0, backend);
}

#[test]
fn test_try_new_rejects_zero_backend_dim() {
    let (cfg, _, s) = matching_tiny();
    let t = MockTransformer {
        dim: 0,
        max_seq: cfg.transformer.max_seq_len,
    };
    let err = construction_error(t, s, cfg.clone());
    assert_config_mismatch(
        err,
        HybridError::FIELD_TRANSFORMER_DIM,
        cfg.transformer.dim,
        0,
    );
}

#[test]
fn test_try_new_rejects_zero_backend_max_seq_len() {
    let (cfg, _, s) = matching_tiny();
    let t = MockTransformer {
        dim: cfg.transformer.dim,
        max_seq: 0,
    };
    let err = construction_error(t, s, cfg.clone());
    assert_config_mismatch(
        err,
        HybridError::FIELD_TRANSFORMER_MAX_SEQ_LEN,
        cfg.transformer.max_seq_len,
        0,
    );
}

#[test]
fn test_try_new_rejects_zero_backend_channels() {
    let (cfg, t, _) = matching_tiny();
    let s = MockSnn::new(0);
    let err = construction_error(t, s, cfg.clone());
    assert_config_mismatch(
        err,
        HybridError::FIELD_SNN_INPUT_CHANNELS,
        cfg.snn_input_channels,
        0,
    );
}

#[test]
fn test_try_new_rejects_both_zero_transformer_dim() {
    let (mut cfg, _, s) = matching_tiny();
    cfg.transformer.dim = 0;
    let t = MockTransformer {
        dim: 0,
        max_seq: cfg.transformer.max_seq_len,
    };
    let err = construction_error(t, s, cfg);
    assert_config_mismatch(err, HybridError::FIELD_TRANSFORMER_DIM, 0, 0);
}

#[test]
fn test_try_new_accepts_tiny_config() {
    let (cfg, t, s) = matching_tiny();
    let net = HybridNetwork::try_new(t, s, cfg).expect("tiny config should construct");
    assert_eq!(net.config().transformer.dim, 128);
    assert_eq!(net.config().transformer.max_seq_len, 64);
    assert_eq!(net.config().snn_input_channels, 64);
    assert_eq!(net.transformer.dim(), 128);
    assert_eq!(net.snn.num_channels(), 64);
}

fn scripted_forward(
    hidden: Tensor,
    dim: usize,
    channels: usize,
    tokens: &[u32],
) -> (
    HybridNetwork<ScriptedTransformer, SpySnn>,
    hybrid_fusion::Result<HybridOutput>,
) {
    let cfg = HybridConfig::tiny();
    let mut net = HybridNetwork::new(
        ScriptedTransformer {
            dim,
            max_seq: cfg.transformer.max_seq_len,
            hidden,
        },
        SpySnn::new(channels),
        cfg,
    );
    let result = net.forward(tokens, None);
    (net, result)
}

#[test]
fn public_forward_rejects_rank0_and_does_not_step() {
    let dim = HybridConfig::tiny().transformer.dim;
    let channels = HybridConfig::tiny().snn_input_channels;
    let (net, result) = scripted_forward(Tensor::from_vec(vec![1.0], &[]), dim, channels, &[1, 2]);
    match result.unwrap_err() {
        HybridError::HiddenStateRank { got: 0 } => {}
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(net.snn.step_calls, 0);
    assert_eq!(net.global_step(), 0);
}

#[test]
fn test_try_new_accepts_custom_width() {
    let mut cfg = HybridConfig::tiny();
    cfg.transformer.dim = 16;
    cfg.transformer.max_seq_len = 8;
    cfg.snn_input_channels = 7;
    let t = MockTransformer {
        dim: 16,
        max_seq: 8,
    };
    let s = MockSnn::new(7);
    let mut net = HybridNetwork::try_new(t, s, cfg).expect("custom width should construct");
    let out = net.forward(&[1, 2], None).expect("forward ok");
    assert_eq!(out.embedding.len(), 16);
    assert_eq!(out.stimuli.len(), 7);
}

#[test]
fn test_try_new_does_not_run_inference() {
    let cfg = HybridConfig::tiny();
    let t = CountingTransformer {
        dim: cfg.transformer.dim,
        max_seq: cfg.transformer.max_seq_len,
        hidden_calls: Cell::new(0),
        param_calls: Cell::new(0),
    };
    let s = CountingSnn {
        channels: cfg.snn_input_channels,
        step_calls: Cell::new(0),
    };
    let net = HybridNetwork::try_new(t, s, cfg).expect("valid tiny");
    assert_eq!(
        net.transformer.hidden_calls.get(),
        0,
        "try_new must not call Transformer::hidden_states"
    );
    assert_eq!(
        net.transformer.param_calls.get(),
        0,
        "try_new must not call Transformer::param_count"
    );
    assert_eq!(
        net.snn.step_calls.get(),
        0,
        "try_new must not call SpikingNetwork::step"
    );
}

#[test]
fn test_new_skips_validation_compatibility() {
    // HybridNetwork::new remains an unvalidated pre-1.0 compatibility path.
    let cfg = HybridConfig::tiny();
    let mock_channels = cfg.snn_input_channels.saturating_add(13).max(1);
    assert_ne!(mock_channels, cfg.snn_input_channels);
    let t = MockTransformer {
        dim: cfg.transformer.dim,
        max_seq: cfg.transformer.max_seq_len,
    };
    let s = MockSnn::new(mock_channels);
    let net = HybridNetwork::new(t, s, cfg);
    assert_ne!(net.snn.num_channels(), net.config().snn_input_channels);
}

#[test]
fn public_forward_rejects_rank1_and_rank3() {
    let dim = HybridConfig::tiny().transformer.dim;
    let channels = HybridConfig::tiny().snn_input_channels;

    let (net, result) =
        scripted_forward(Tensor::from_vec(vec![0.0; 3], &[3]), dim, channels, &[1, 2]);
    match result.unwrap_err() {
        HybridError::HiddenStateDim { expected, got: 3 } if expected == dim => {}
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(net.snn.step_calls, 0);

    let (net, result) = scripted_forward(
        Tensor::from_vec(vec![0.0; 8], &[2, 2, 2]),
        dim,
        channels,
        &[1, 2],
    );
    match result.unwrap_err() {
        HybridError::HiddenStateRank { got: 3 } => {}
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(net.snn.step_calls, 0);
    assert_eq!(net.global_step(), 0);
}

#[test]
fn public_forward_rejects_seq_len_and_non_finite() {
    let dim = HybridConfig::tiny().transformer.dim;
    let channels = HybridConfig::tiny().snn_input_channels;
    let data: Vec<f32> = (0..3 * dim).map(|i| i as f32 * 0.01).collect();
    let (net, result) = scripted_forward(Tensor::from_vec(data, &[3, dim]), dim, channels, &[1, 2]);
    match result.unwrap_err() {
        HybridError::HiddenStateSeqLen {
            expected: 2,
            got: 3,
        } => {}
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(net.snn.step_calls, 0);

    let mut hidden = Tensor::from_vec(vec![0.1; 2 * dim], &[2, dim]);
    hidden.data_mut()[0] = f32::NAN;
    let (net, result) = scripted_forward(hidden, dim, channels, &[1, 2]);
    match result.unwrap_err() {
        HybridError::NonFinite {
            stage: ForwardValueStage::HiddenState,
            index: 0,
        } => {}
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(net.snn.step_calls, 0);

    let mut hidden = Tensor::from_vec(vec![0.1; 2 * dim], &[2, dim]);
    hidden.data_mut()[1] = f32::INFINITY;
    let (net, result) = scripted_forward(hidden, dim, channels, &[1, 2]);
    match result.unwrap_err() {
        HybridError::NonFinite {
            stage: ForwardValueStage::HiddenState,
            index: 1,
        } => {}
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(net.snn.step_calls, 0);

    let mut hidden = Tensor::from_vec(vec![0.1; 2 * dim], &[2, dim]);
    hidden.data_mut()[2] = f32::NEG_INFINITY;
    let (net, result) = scripted_forward(hidden, dim, channels, &[1, 2]);
    match result.unwrap_err() {
        HybridError::NonFinite {
            stage: ForwardValueStage::HiddenState,
            index: 2,
        } => {}
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(net.snn.step_calls, 0);
}

#[test]
fn public_forward_rejects_empty_and_inconsistent_storage() {
    let dim = HybridConfig::tiny().transformer.dim;
    let channels = HybridConfig::tiny().snn_input_channels;
    let empty: Tensor = serde_json::from_value(serde_json::json!({
        "data": [],
        "shape": [2, dim],
    }))
    .unwrap();
    let (net, result) = scripted_forward(empty, dim, channels, &[1, 2]);
    match result.unwrap_err() {
        HybridError::HiddenStateDataLen { expected, got: 0 } if expected == 2 * dim => {}
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(net.snn.step_calls, 0);

    let inconsistent: Tensor = serde_json::from_value(serde_json::json!({
        "data": [0.0, 1.0],
        "shape": [2, dim],
    }))
    .unwrap();
    let (net, result) = scripted_forward(inconsistent, dim, channels, &[1, 2]);
    match result.unwrap_err() {
        HybridError::HiddenStateDataLen { expected, got: 2 } if expected == 2 * dim => {}
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(net.snn.step_calls, 0);
    assert_eq!(net.global_step(), 0);
}

#[test]
fn public_forward_rejects_zero_snn_channels() {
    let dim = HybridConfig::tiny().transformer.dim;
    let hidden = Tensor::from_vec(vec![0.1; 2 * dim], &[2, dim]);
    let (net, result) = scripted_forward(hidden, dim, 0, &[1, 2]);
    match result.unwrap_err() {
        HybridError::ZeroSnnChannels => {}
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(net.snn.step_calls, 0);
    assert_eq!(net.global_step(), 0);
}

#[test]
fn public_forward_valid_rank2_regression() {
    let mut net = build_network();
    let tokens = [1u32, 2, 3, 4];
    let out = net.forward(&tokens, None).unwrap();
    assert_eq!(out.embedding.len(), net.transformer.dim());
    assert_eq!(out.stimuli.len(), net.snn.num_channels());
    assert_eq!(out.global_step, 1);
    let dim = net.transformer.dim();
    let expected_0 = 0.01 * dim as f32 * (tokens.len() - 1) as f32 / 2.0;
    assert!((out.embedding[0] - expected_0).abs() < 1e-5);
}
