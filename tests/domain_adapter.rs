// SPDX-License-Identifier: MIT OR Apache-2.0

//! `DomainAdapter` contract tests from a downstream crate's point of view
//! (public API only): the reference adapter, a stateful third-party adapter,
//! and parity with the forward host's existing projector path.

use hybrid_fusion::adapter::conformance::{
    ConformanceDims, assert_domain_adapter_contract, check_domain_adapter_contract_with,
};
use hybrid_fusion::{
    DomainAdapter, ExpertRouteOutput, ExpertRouter, HybridConfig, HybridError, HybridNetwork,
    NeuroModulators, PoolResizeTanhAdapter, ProjectionMode, Result, ReverseHybridPath,
    SpikeActivity, SpikeEncoding, SpikeEncodingKind, SpikingNetwork, Tensor, Transformer,
    decode_activity_checked, encode_stimuli_checked, validate_decode_input, validate_embedding,
    validate_encode_input, validate_stimuli,
};

/// Stateful adapter: stimuli are an exponential moving average of the
/// reference encoding across calls, so output depends on call history. Decode
/// is stateless and delegated. State is committed only after validation, as
/// the failure-atomicity clause requires.
struct SmoothingAdapter {
    inner: PoolResizeTanhAdapter,
    memory: Vec<f32>,
}

impl SmoothingAdapter {
    fn new() -> Self {
        Self {
            inner: PoolResizeTanhAdapter::new(ProjectionMode::RateSum),
            memory: Vec::new(),
        }
    }
}

impl DomainAdapter for SmoothingAdapter {
    fn encode_stimuli(&mut self, hidden: &Tensor, snn_width: usize) -> Result<Vec<f32>> {
        validate_encode_input(hidden, snn_width)?;
        let fresh = self.inner.encode_stimuli(hidden, snn_width)?;
        let previous = if self.memory.len() == snn_width {
            self.memory.clone()
        } else {
            vec![0.0; snn_width]
        };
        // Convex combination of two values in [-1, 1] stays in [-1, 1].
        let smoothed: Vec<f32> = fresh
            .iter()
            .zip(&previous)
            .map(|(f, p)| 0.5 * f + 0.5 * p)
            .collect();
        validate_stimuli(&smoothed, snn_width)?;
        self.memory.clone_from(&smoothed);
        Ok(smoothed)
    }

    fn decode_activity(
        &mut self,
        activity: &SpikeActivity,
        num_neurons: usize,
        embed_dim: usize,
    ) -> Result<Vec<f32>> {
        validate_decode_input(activity, num_neurons, embed_dim)?;
        let embedding = self
            .inner
            .decode_activity(activity, num_neurons, embed_dim)?;
        validate_embedding(&embedding, embed_dim)?;
        Ok(embedding)
    }

    fn reset(&mut self) {
        self.memory.clear();
    }

    fn encoding(&self) -> SpikeEncoding {
        SpikeEncoding::new(
            SpikeEncodingKind::Other("smoothed".into()),
            1,
            Some((-1.0, 1.0)),
        )
    }
}

#[test]
fn reference_adapter_conforms_in_every_projection_mode() {
    for mode in [
        ProjectionMode::RateSum,
        ProjectionMode::TemporalHistogram,
        ProjectionMode::MembraneSnapshot,
        ProjectionMode::SpikingTernary,
    ] {
        assert_domain_adapter_contract(|| PoolResizeTanhAdapter::new(mode));
    }
}

#[test]
fn stateful_third_party_adapter_conforms() {
    assert_domain_adapter_contract(SmoothingAdapter::new);
    check_domain_adapter_contract_with(
        SmoothingAdapter::new,
        ConformanceDims {
            snn_width: 1,
            num_neurons: 1,
            embed_dim: 1,
        },
    )
    .expect("minimal dims");
}

#[test]
fn smoothing_adapter_really_is_stateful() {
    let hidden = Tensor::from_vec(vec![2.0; 4], &[4]);
    let mut adapter = SmoothingAdapter::new();
    let first = encode_stimuli_checked(&mut adapter, &hidden, 4).unwrap();
    let second = encode_stimuli_checked(&mut adapter, &hidden, 4).unwrap();
    assert_ne!(first, second, "history must change the output");
    adapter.reset();
    assert_eq!(
        encode_stimuli_checked(&mut adapter, &hidden, 4).unwrap(),
        first
    );

    // A rejected call must not advance the state.
    let bad = Tensor::from_vec(vec![f32::NAN; 4], &[4]);
    encode_stimuli_checked(&mut adapter, &hidden, 4).unwrap();
    assert!(encode_stimuli_checked(&mut adapter, &bad, 4).is_err());
    let mut twin = SmoothingAdapter::new();
    encode_stimuli_checked(&mut twin, &hidden, 4).unwrap();
    encode_stimuli_checked(&mut twin, &hidden, 4).unwrap();
    assert_eq!(
        encode_stimuli_checked(&mut adapter, &hidden, 4).unwrap(),
        encode_stimuli_checked(&mut twin, &hidden, 4).unwrap()
    );
}

#[test]
fn hostile_inputs_return_errors_instead_of_panicking() {
    let mut adapter = PoolResizeTanhAdapter::default();
    let rank3 = Tensor::from_vec(vec![0.0; 8], &[2, 2, 2]);
    assert!(matches!(
        encode_stimuli_checked(&mut adapter, &rank3, 4),
        Err(HybridError::HiddenStateRank { got: 3 })
    ));
    let inconsistent: Tensor =
        serde_json::from_str(r#"{"data":[0.1,0.2,0.3],"shape":[2,3]}"#).unwrap();
    for result in [
        adapter.encode_stimuli(&inconsistent, 4),
        encode_stimuli_checked(&mut adapter, &inconsistent, 4),
    ] {
        assert!(matches!(
            result,
            Err(HybridError::HiddenStateDataLen {
                expected: 6,
                got: 3
            })
        ));
    }
    let activity = SpikeActivity {
        spike_train: vec![vec![9]],
        potentials: vec![0.0; 4],
        iz_potentials: vec![],
    };
    assert!(decode_activity_checked(&mut adapter, &activity, 4, 3).is_err());
    assert!(adapter.decode_activity(&activity, 4, 3).is_err());
}

struct PatternTransformer {
    dim: usize,
}

impl PatternTransformer {
    fn hidden(&self, seq: usize) -> Tensor {
        let data = (0..seq * self.dim)
            .map(|i| (i as f32 * 0.113).sin() * 2.5)
            .collect();
        Tensor::from_vec(data, &[seq, self.dim])
    }
}

impl Transformer for PatternTransformer {
    fn hidden_states(&self, token_ids: &[u32]) -> Tensor {
        self.hidden(token_ids.len())
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn max_seq_len(&self) -> usize {
        64
    }
    fn param_count(&self) -> usize {
        0
    }
}

struct SilentSnn {
    channels: usize,
}

impl SpikingNetwork for SilentSnn {
    fn step(&mut self, _stimuli: &[f32], _modulators: &NeuroModulators) -> Result<Vec<usize>> {
        Ok(Vec::new())
    }
    fn num_channels(&self) -> usize {
        self.channels
    }
}

/// The reference adapter is the forward host's existing path behind the
/// contract: stimuli match `HybridNetwork::forward` bit for bit.
#[test]
fn reference_encode_is_bitwise_identical_to_the_forward_host() {
    let config = HybridConfig::tiny();
    let dim = config.transformer.dim;
    let channels = config.snn_input_channels;
    let transformer = PatternTransformer { dim };
    let tokens = [3u32, 1, 4, 1, 5];
    let hidden = transformer.hidden(tokens.len());
    let mut net = HybridNetwork::try_new(transformer, SilentSnn { channels }, config).unwrap();
    let forward = net.forward(&tokens, None).unwrap();

    let mut adapter = PoolResizeTanhAdapter::default();
    let stimuli = encode_stimuli_checked(&mut adapter, &hidden, channels).unwrap();
    assert_eq!(
        stimuli.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        forward
            .stimuli
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}

// ── Host wiring (RM-1811 follow-up) ─────────────────────────────────────────

/// Adapter emitting a fixed stimulus vector, no matter the input.
struct ConstantAdapter {
    value: f32,
}

impl DomainAdapter for ConstantAdapter {
    fn encode_stimuli(&mut self, hidden: &Tensor, snn_width: usize) -> Result<Vec<f32>> {
        validate_encode_input(hidden, snn_width)?;
        let out = vec![self.value; snn_width];
        validate_stimuli(&out, snn_width)?;
        Ok(out)
    }

    fn decode_activity(
        &mut self,
        activity: &SpikeActivity,
        num_neurons: usize,
        embed_dim: usize,
    ) -> Result<Vec<f32>> {
        validate_decode_input(activity, num_neurons, embed_dim)?;
        let out = vec![self.value; embed_dim];
        validate_embedding(&out, embed_dim)?;
        Ok(out)
    }
}

/// Adapter whose hooks always fail, to prove errors propagate.
struct FailingAdapter;

impl DomainAdapter for FailingAdapter {
    fn encode_stimuli(&mut self, _hidden: &Tensor, _snn_width: usize) -> Result<Vec<f32>> {
        Err(HybridError::InvalidConfig(
            "FailingAdapter::encode_stimuli".into(),
        ))
    }

    fn decode_activity(
        &mut self,
        _activity: &SpikeActivity,
        _num_neurons: usize,
        _embed_dim: usize,
    ) -> Result<Vec<f32>> {
        Err(HybridError::InvalidConfig(
            "FailingAdapter::decode_activity".into(),
        ))
    }
}

/// SNN that counts how often it is stepped.
struct CountingSnn {
    channels: usize,
    steps: u32,
}

impl SpikingNetwork for CountingSnn {
    fn step(&mut self, _stimuli: &[f32], _modulators: &NeuroModulators) -> Result<Vec<usize>> {
        self.steps += 1;
        Ok(Vec::new())
    }
    fn num_channels(&self) -> usize {
        self.channels
    }
}

/// `forward` produces stimuli through the injected adapter, not the in-line
/// projector path.
#[test]
fn hybrid_network_forward_uses_the_injected_adapter() {
    let config = HybridConfig::tiny();
    let channels = config.snn_input_channels;
    let mut net = HybridNetwork::try_with_adapter(
        PatternTransformer {
            dim: config.transformer.dim,
        },
        CountingSnn { channels, steps: 0 },
        config,
        Box::new(ConstantAdapter { value: 0.25 }),
    )
    .unwrap();
    let out = net.forward(&[1, 2, 3], None).unwrap();
    assert_eq!(out.stimuli, vec![0.25; channels]);
    assert_eq!(net.snn.steps, 1);
    assert_eq!(net.global_step(), 1);
}

/// An adapter error propagates: the SNN is never stepped and `global_step`
/// does not advance.
#[test]
fn hybrid_network_forward_propagates_adapter_errors_without_stepping() {
    let config = HybridConfig::tiny();
    let channels = config.snn_input_channels;
    let mut net = HybridNetwork::try_with_adapter(
        PatternTransformer {
            dim: config.transformer.dim,
        },
        CountingSnn { channels, steps: 0 },
        config,
        Box::new(FailingAdapter),
    )
    .unwrap();
    match net.forward(&[1, 2, 3], None) {
        Err(HybridError::InvalidConfig(msg)) => {
            assert!(msg.contains("FailingAdapter"), "{msg}")
        }
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(net.snn.steps, 0);
    assert_eq!(net.global_step(), 0);
}

/// `reset()` clears injected adapter state, so a stateful adapter replays its
/// fresh outputs after reset.
#[test]
fn hybrid_network_reset_clears_adapter_state() {
    let config = HybridConfig::tiny();
    let channels = config.snn_input_channels;
    let mut net = HybridNetwork::try_with_adapter(
        PatternTransformer {
            dim: config.transformer.dim,
        },
        CountingSnn { channels, steps: 0 },
        config,
        Box::new(SmoothingAdapter::new()),
    )
    .unwrap();
    let first = net.forward(&[4, 2], None).unwrap().stimuli;
    let second = net.forward(&[4, 2], None).unwrap().stimuli;
    assert_ne!(first, second, "smoothing adapter must be stateful");
    net.reset().unwrap();
    assert_eq!(
        net.forward(&[4, 2], None).unwrap().stimuli,
        first,
        "after reset the adapter must replay its first output"
    );
}

/// Router that records the embedding it was given.
#[derive(Debug, Default)]
struct RecordingRouter {
    last: Vec<f32>,
}

impl ExpertRouter for RecordingRouter {
    fn num_experts(&self) -> usize {
        2
    }
    fn top_k(&self) -> usize {
        1
    }
    fn route(&mut self, embedding: &[f32]) -> Result<ExpertRouteOutput> {
        self.last = embedding.to_vec();
        Ok(ExpertRouteOutput {
            expert_weights: vec![0.5, 0.5],
            selected_experts: vec![0],
            routing_entropy: None,
        })
    }
}

/// `forward_activity` produces the embedding through the injected adapter and
/// hands it to the router.
#[test]
fn reverse_path_uses_the_injected_adapter() {
    let act = SpikeActivity::from_fired(&[0], 4).unwrap();
    let mut path = ReverseHybridPath::with_adapter(
        ProjectionMode::RateSum,
        4,
        6,
        RecordingRouter::default(),
        Box::new(ConstantAdapter { value: 0.75 }),
    )
    .unwrap();
    let out = path.forward_activity(&act).unwrap();
    assert_eq!(out.embedding, vec![0.75; 6]);
    assert_eq!(path.router().last, vec![0.75; 6]);
    assert_eq!(path.mode(), ProjectionMode::RateSum);
}

/// Decode errors propagate before routing, without advancing `global_step`.
#[test]
fn reverse_path_propagates_adapter_errors() {
    let act = SpikeActivity::from_fired(&[0], 4).unwrap();
    let mut path = ReverseHybridPath::with_adapter(
        ProjectionMode::RateSum,
        4,
        6,
        RecordingRouter::default(),
        Box::new(FailingAdapter),
    )
    .unwrap();
    assert!(path.forward_activity(&act).is_err());
    assert_eq!(path.global_step(), 0);
    assert!(path.router().last.is_empty(), "router must not be reached");
}

/// `reset()` clears injected decode-adapter state (a decode that depends on
/// call count must replay its fresh output).
#[test]
fn reverse_path_reset_clears_adapter_state() {
    struct CountingDecoder {
        calls: u32,
    }
    impl DomainAdapter for CountingDecoder {
        fn encode_stimuli(&mut self, hidden: &Tensor, snn_width: usize) -> Result<Vec<f32>> {
            validate_encode_input(hidden, snn_width)?;
            Ok(vec![0.0; snn_width])
        }
        fn decode_activity(
            &mut self,
            activity: &SpikeActivity,
            num_neurons: usize,
            embed_dim: usize,
        ) -> Result<Vec<f32>> {
            validate_decode_input(activity, num_neurons, embed_dim)?;
            self.calls += 1;
            Ok(vec![self.calls as f32; embed_dim])
        }
        fn reset(&mut self) {
            self.calls = 0;
        }
    }

    let act = SpikeActivity::from_fired(&[0], 4).unwrap();
    let mut path = ReverseHybridPath::with_adapter(
        ProjectionMode::RateSum,
        4,
        6,
        RecordingRouter::default(),
        Box::new(CountingDecoder { calls: 0 }),
    )
    .unwrap();
    assert_eq!(path.forward_activity(&act).unwrap().embedding, vec![1.0; 6]);
    assert_eq!(path.forward_activity(&act).unwrap().embedding, vec![2.0; 6]);
    path.reset();
    assert_eq!(
        path.forward_activity(&act).unwrap().embedding,
        vec![1.0; 6],
        "after reset the adapter must replay its first output"
    );
}
