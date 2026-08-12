// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration tests for `HybridNetwork` using mock backends.
//!
//! These tests exercise only the public API surface — no `pub(crate)` or internal access —
//! and serve as usage examples for consumers of the crate.
//!
//! Note: `HybridNetwork::{transformer, snn}` are public fields and are part of the public
//! surface; tests prefer `config()` / output fields when possible.

use hybrid_fusion::{
    HybridConfig, HybridError, HybridNetwork, HybridOutput, NeuroModulators, SpikingNetwork,
    Tensor, Transformer,
};

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
/// the public forward API (InvalidConfig) rather than silently projected.
#[test]
fn test_forward_rejects_hidden_dim_mismatch() {
    let cfg = HybridConfig::tiny();
    let t = MismatchedDimTransformer {
        reported_dim: cfg.transformer.dim,
        actual_dim: cfg.transformer.dim + 8,
        max_seq: cfg.transformer.max_seq_len,
    };
    let s = MockSnn::new(cfg.snn_input_channels);
    let mut net = HybridNetwork::new(t, s, cfg);

    let err = net
        .forward(&[1, 2, 3], None)
        .expect_err("mismatched hidden dim should fail");
    // Assert the public error variant only — do not pin free-form message text.
    match err {
        HybridError::InvalidConfig(_) => {}
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}
