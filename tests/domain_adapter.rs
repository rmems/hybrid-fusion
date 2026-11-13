// SPDX-License-Identifier: MIT OR Apache-2.0

//! `DomainAdapter` contract tests from a downstream crate's point of view
//! (public API only): the reference adapter, a stateful third-party adapter,
//! and parity with the forward host's existing projector path.

use hybrid_fusion::adapter::conformance::{
    ConformanceDims, assert_domain_adapter_contract, check_domain_adapter_contract_with,
};
use hybrid_fusion::{
    DomainAdapter, HybridConfig, HybridError, HybridNetwork, NeuroModulators,
    PoolResizeTanhAdapter, ProjectionMode, Result, SpikeActivity, SpikeEncoding, SpikeEncodingKind,
    SpikingNetwork, Tensor, Transformer, validate_decode_input, validate_embedding,
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
    let first = adapter.encode_stimuli_checked(&hidden, 4).unwrap();
    let second = adapter.encode_stimuli_checked(&hidden, 4).unwrap();
    assert_ne!(first, second, "history must change the output");
    adapter.reset();
    assert_eq!(adapter.encode_stimuli_checked(&hidden, 4).unwrap(), first);

    // A rejected call must not advance the state.
    let bad = Tensor::from_vec(vec![f32::NAN; 4], &[4]);
    adapter.encode_stimuli_checked(&hidden, 4).unwrap();
    assert!(adapter.encode_stimuli_checked(&bad, 4).is_err());
    let mut twin = SmoothingAdapter::new();
    twin.encode_stimuli_checked(&hidden, 4).unwrap();
    twin.encode_stimuli_checked(&hidden, 4).unwrap();
    assert_eq!(
        adapter.encode_stimuli_checked(&hidden, 4).unwrap(),
        twin.encode_stimuli_checked(&hidden, 4).unwrap()
    );
}

#[test]
fn hostile_inputs_return_errors_instead_of_panicking() {
    let mut adapter = PoolResizeTanhAdapter::default();
    let rank3 = Tensor::from_vec(vec![0.0; 8], &[2, 2, 2]);
    assert!(matches!(
        adapter.encode_stimuli_checked(&rank3, 4),
        Err(HybridError::HiddenStateRank { got: 3 })
    ));
    let inconsistent: Tensor =
        serde_json::from_str(r#"{"data":[0.1,0.2,0.3],"shape":[2,3]}"#).unwrap();
    for result in [
        adapter.encode_stimuli(&inconsistent, 4),
        adapter.encode_stimuli_checked(&inconsistent, 4),
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
    assert!(adapter.decode_activity_checked(&activity, 4, 3).is_err());
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
    let stimuli = adapter.encode_stimuli_checked(&hidden, channels).unwrap();
    assert_eq!(
        stimuli.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        forward
            .stimuli
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}
