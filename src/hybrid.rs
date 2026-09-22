// SPDX-License-Identifier: MIT OR Apache-2.0

//! ANN → SNN forward-path host: transformer hidden states → bounded SNN stimuli.

use crate::error::{ForwardValueStage, HybridError, Result};
use crate::projector;
use crate::telemetry;
use crate::tensor::Tensor;
use crate::traits::{NeuroModulators, SpikingNetwork, Transformer};
use crate::types::{HybridConfig, HybridOutput};
use crate::validate;

/// Generic orchestrator over any [`Transformer`] + [`SpikingNetwork`].
///
/// Prefer [`Self::try_new`] so transformer dimensions, maximum sequence length,
/// and SNN input width are checked against the injected backends before the
/// first [`Self::forward`] call. [`Self::new`] remains as an unvalidated
/// pre-1.0 compatibility constructor.
pub struct HybridNetwork<T: Transformer, S: SpikingNetwork> {
    pub transformer: T,
    pub snn: S,
    config: HybridConfig,
    global_step: u64,
}

impl<T: Transformer, S: SpikingNetwork> HybridNetwork<T, S> {
    /// Construct without validating `config` against backend capabilities.
    ///
    /// Prefer [`Self::try_new`], which rejects zero or disagreeing transformer
    /// dimensions, maximum sequence lengths, and SNN channel counts.
    ///
    /// This infallible constructor is retained as a pre-1.0 compatibility path
    /// so existing callers keep compiling without handling [`Result`]. It does
    /// not call [`Transformer::hidden_states`] or [`SpikingNetwork::step`].
    pub fn new(transformer: T, snn: S, config: HybridConfig) -> Self {
        Self {
            transformer,
            snn,
            config,
            global_step: 0,
        }
    }

    /// Construct after proving `config` agrees with backend-reported capabilities.
    ///
    /// Validates, without running inference or mutating either backend:
    /// - `config.transformer.dim` against [`Transformer::dim`]
    /// - `config.transformer.max_seq_len` against [`Transformer::max_seq_len`]
    /// - `config.snn_input_channels` against [`SpikingNetwork::num_channels`]
    ///
    /// Each check requires both sides to be non-zero and equal. Failures return
    /// [`HybridError::ConfigMismatch`] naming the field plus configured vs
    /// backend values. The fields are
    /// [`HybridError::FIELD_TRANSFORMER_DIM`],
    /// [`HybridError::FIELD_TRANSFORMER_MAX_SEQ_LEN`], and
    /// [`HybridError::FIELD_SNN_INPUT_CHANNELS`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hybrid_fusion::{
    ///     HybridConfig, HybridNetwork, NeuroModulators, Result, SpikingNetwork, Tensor,
    ///     Transformer,
    /// };
    ///
    /// struct TinyTransformer {
    ///     dim: usize,
    ///     max_seq_len: usize,
    /// }
    ///
    /// impl Transformer for TinyTransformer {
    ///     fn hidden_states(&self, token_ids: &[u32]) -> Tensor {
    ///         let seq = token_ids.len();
    ///         Tensor::from_vec(vec![0.1; seq * self.dim], &[seq, self.dim])
    ///     }
    ///     fn dim(&self) -> usize {
    ///         self.dim
    ///     }
    ///     fn max_seq_len(&self) -> usize {
    ///         self.max_seq_len
    ///     }
    ///     fn param_count(&self) -> usize {
    ///         0
    ///     }
    /// }
    ///
    /// struct TinySnn {
    ///     channels: usize,
    /// }
    ///
    /// impl SpikingNetwork for TinySnn {
    ///     fn step(
    ///         &mut self,
    ///         _stimuli: &[f32],
    ///         _modulators: &NeuroModulators,
    ///     ) -> Result<Vec<usize>> {
    ///         Ok(Vec::new())
    ///     }
    ///     fn num_channels(&self) -> usize {
    ///         self.channels
    ///     }
    /// }
    ///
    /// let config = HybridConfig::tiny();
    /// let transformer = TinyTransformer {
    ///     dim: config.transformer.dim,
    ///     max_seq_len: config.transformer.max_seq_len,
    /// };
    /// let snn = TinySnn {
    ///     channels: config.snn_input_channels,
    /// };
    /// let mut net = HybridNetwork::try_new(transformer, snn, config)?;
    /// let out = net.forward(&[1u32, 2, 3, 4], None)?;
    /// assert_eq!(out.embedding.len(), 128);
    /// # Ok::<(), hybrid_fusion::HybridError>(())
    /// ```
    pub fn try_new(transformer: T, snn: S, config: HybridConfig) -> Result<Self> {
        validate_construction(&transformer, &snn, &config)?;
        Ok(Self::new(transformer, snn, config))
    }

    pub fn forward(
        &mut self,
        token_ids: &[u32],
        modulators: Option<NeuroModulators>,
    ) -> Result<HybridOutput> {
        // Caller-side validation and backend-contract errors are returned
        // without Sentry capture so routine bad requests / malformed hidden
        // states do not flood the error stream / quota.
        if token_ids.is_empty() {
            return Err(HybridError::InputLengthMismatch {
                expected: 1,
                got: 0,
            });
        }
        if token_ids.len() > self.transformer.max_seq_len() {
            return Err(HybridError::InputLengthMismatch {
                expected: self.transformer.max_seq_len(),
                got: token_ids.len(),
            });
        }

        // Host SNN width is a precondition of projection; reject before the
        // transformer runs so a zero-channel backend cannot be stepped.
        let snn_width = self.snn.num_channels();
        validate::validate_snn_width(snn_width)?;

        let hidden = self.transformer.hidden_states(token_ids);
        validate::validate_hidden_state(&hidden, token_ids.len(), self.transformer.dim())?;

        let embedding = pool_embedding(&hidden, self.transformer.dim());
        validate::validate_finite(&embedding, ForwardValueStage::Embedding)?;

        let stimuli = projector::embed_to_stimuli_with_width(&hidden, snn_width);
        validate::validate_finite(&stimuli, ForwardValueStage::Stimuli)?;

        let modulators = modulators.unwrap_or_default();
        // Backend/runtime failures are reported when the `sentry` feature is on.
        let fired_neurons = match self.snn.step(&stimuli, &modulators) {
            Ok(fired) => fired,
            Err(err) => {
                telemetry::capture_error(&err);
                return Err(err);
            }
        };

        self.global_step = self.global_step.saturating_add(1);

        Ok(HybridOutput {
            embedding,
            stimuli,
            fired_neurons,
            global_step: self.global_step,
            // Reverse-path MoE fields stay unset on the ANN→SNN forward pass.
            // Use ReverseHybridPath::forward_activity for activity → MoE.
            expert_weights: None,
            selected_experts: None,
            routing_entropy: None,
        })
    }

    pub fn config(&self) -> &HybridConfig {
        &self.config
    }

    pub fn global_step(&self) -> u64 {
        self.global_step
    }

    pub fn reset(&mut self) {
        self.global_step = 0;
    }
}

/// Compare `config` against trait-reported backend sizes. Reads only
/// [`Transformer::dim`], [`Transformer::max_seq_len`], and
/// [`SpikingNetwork::num_channels`] — never hidden states or an SNN step.
fn validate_construction<T: Transformer, S: SpikingNetwork>(
    transformer: &T,
    snn: &S,
    config: &HybridConfig,
) -> Result<()> {
    check_capacity(
        HybridError::FIELD_TRANSFORMER_DIM,
        config.transformer.dim,
        transformer.dim(),
    )?;
    check_capacity(
        HybridError::FIELD_TRANSFORMER_MAX_SEQ_LEN,
        config.transformer.max_seq_len,
        transformer.max_seq_len(),
    )?;
    check_capacity(
        HybridError::FIELD_SNN_INPUT_CHANNELS,
        config.snn_input_channels,
        snn.num_channels(),
    )?;
    Ok(())
}

fn check_capacity(field: &'static str, configured: usize, backend: usize) -> Result<()> {
    // Both-zero would otherwise look like a match (`configured == backend`).
    let both_positive_and_equal = configured > 0 && configured == backend;
    if !both_positive_and_equal {
        return Err(HybridError::config_mismatch(field, configured, backend));
    }
    Ok(())
}

fn pool_embedding(hidden: &Tensor, dim: usize) -> Vec<f32> {
    if hidden.ndim() == 1 {
        return hidden.data().to_vec();
    }
    if hidden.ndim() == 2 {
        let shape = hidden.shape();
        let seq = shape[0].max(1);
        let hdim = shape[1];
        let take = dim.min(hdim);
        let data = hidden.data();
        let mut pooled = vec![0.0f32; take];
        for t in 0..seq {
            let row = &data[t * hdim..(t + 1) * hdim];
            for (i, v) in row.iter().take(take).enumerate() {
                pooled[i] += *v;
            }
        }
        let inv = 1.0 / seq as f32;
        for v in &mut pooled {
            *v *= inv;
        }
        return pooled;
    }
    hidden.data().iter().copied().take(dim).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use crate::tensor::Tensor;
    use crate::traits::{NeuroModulators, SpikingNetwork, Transformer};
    use crate::types::HybridConfig;

    struct MockTransformer {
        dim: usize,
        max_seq: usize,
    }

    impl Transformer for MockTransformer {
        fn hidden_states(&self, token_ids: &[u32]) -> Tensor {
            let seq = token_ids.len();
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
    }

    impl SpikingNetwork for MockSnn {
        fn step(&mut self, stimuli: &[f32], _modulators: &NeuroModulators) -> Result<Vec<usize>> {
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

    fn build_network() -> HybridNetwork<MockTransformer, MockSnn> {
        let cfg = HybridConfig::tiny();
        let t = MockTransformer {
            dim: cfg.transformer.dim,
            max_seq: cfg.transformer.max_seq_len,
        };
        let s = MockSnn {
            channels: cfg.snn_input_channels,
        };
        HybridNetwork::new(t, s, cfg)
    }

    #[test]
    fn test_forward_shape_and_bounds() {
        let mut net = build_network();
        let out = net.forward(&[1, 2, 3, 4], None).expect("forward ok");
        assert_eq!(out.embedding.len(), 128);
        assert_eq!(out.stimuli.len(), 64);
        for v in &out.stimuli {
            assert!(v.abs() <= 1.0);
        }
        assert_eq!(out.global_step, 1);
    }

    #[test]
    fn test_forward_rejects_empty() {
        let mut net = build_network();
        assert!(net.forward(&[], None).is_err());
    }

    #[test]
    fn test_forward_rejects_over_long() {
        let mut net = build_network();
        let too_long = vec![0u32; 65];
        assert!(net.forward(&too_long, None).is_err());
    }

    #[test]
    fn test_global_step_increments() {
        let mut net = build_network();
        net.forward(&[0, 1], None).unwrap();
        net.forward(&[0, 1], None).unwrap();
        assert_eq!(net.global_step(), 2);
        net.reset();
        assert_eq!(net.global_step(), 0);
    }

    #[test]
    fn test_snn_width_independent_from_transformer_dim() {
        let mut cfg = HybridConfig::tiny();
        cfg.snn_input_channels = 7;
        let t = MockTransformer {
            dim: cfg.transformer.dim,
            max_seq: cfg.transformer.max_seq_len,
        };
        let s = MockSnn { channels: 7 };
        let mut net = HybridNetwork::try_new(t, s, cfg).expect("custom width is valid");
        let out = net.forward(&[0, 1, 2], None).unwrap();
        assert_eq!(out.stimuli.len(), 7);
        assert_eq!(out.embedding.len(), 128);
    }

    #[test]
    fn test_try_new_accepts_tiny_config() {
        let cfg = HybridConfig::tiny();
        let t = MockTransformer {
            dim: cfg.transformer.dim,
            max_seq: cfg.transformer.max_seq_len,
        };
        let s = MockSnn {
            channels: cfg.snn_input_channels,
        };
        let net = HybridNetwork::try_new(t, s, cfg).expect("tiny config should construct");
        assert_eq!(net.config().transformer.dim, 128);
        assert_eq!(net.config().snn_input_channels, 64);
        assert_eq!(net.global_step(), 0);
    }

    // ── Preflight: spy SNN must not step; global_step stays 0 ───────────────

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

    struct SpySnn {
        channels: usize,
        step_calls: usize,
        state: u32,
    }

    impl SpikingNetwork for SpySnn {
        fn step(&mut self, _stimuli: &[f32], _modulators: &NeuroModulators) -> Result<Vec<usize>> {
            self.step_calls += 1;
            self.state = self.state.wrapping_add(1);
            Ok(Vec::new())
        }
        fn num_channels(&self) -> usize {
            self.channels
        }
    }

    const SPY_STATE: u32 = 7;

    fn scripted_net(
        hidden: Tensor,
        dim: usize,
        channels: usize,
    ) -> HybridNetwork<ScriptedTransformer, SpySnn> {
        let cfg = HybridConfig::tiny();
        HybridNetwork::new(
            ScriptedTransformer {
                dim,
                max_seq: cfg.transformer.max_seq_len,
                hidden,
            },
            SpySnn {
                channels,
                step_calls: 0,
                state: SPY_STATE,
            },
            cfg,
        )
    }

    fn tensor_parts(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        serde_json::from_value(serde_json::json!({
            "data": data,
            "shape": shape,
        }))
        .expect("deserialize Tensor")
    }

    fn assert_preflight_frozen(net: &HybridNetwork<ScriptedTransformer, SpySnn>) {
        assert_eq!(
            net.snn.step_calls, 0,
            "snn.step must not run on preflight error"
        );
        assert_eq!(net.snn.state, SPY_STATE, "SNN state must be unchanged");
        assert_eq!(
            net.global_step(),
            0,
            "global_step must stay 0 on validation error"
        );
    }

    fn tiny_dims() -> (usize, usize) {
        let cfg = HybridConfig::tiny();
        (cfg.transformer.dim, cfg.snn_input_channels)
    }

    fn valid_rank2(seq: usize, dim: usize) -> Tensor {
        let data: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.01).collect();
        Tensor::from_vec(data, &[seq, dim])
    }

    #[test]
    fn preflight_rejects_rank0_without_stepping() {
        let (dim, channels) = tiny_dims();
        let mut net = scripted_net(Tensor::from_vec(vec![1.0], &[]), dim, channels);
        match net.forward(&[1, 2, 3], None).unwrap_err() {
            HybridError::HiddenStateRank { got: 0 } => {}
            other => panic!("unexpected {other:?}"),
        }
        assert_preflight_frozen(&net);
    }

    #[test]
    fn preflight_rejects_rank1_wrong_dim_without_stepping() {
        let (dim, channels) = tiny_dims();
        let mut net = scripted_net(
            Tensor::from_vec(vec![0.1; dim + 1], &[dim + 1]),
            dim,
            channels,
        );
        match net.forward(&[1, 2], None).unwrap_err() {
            HybridError::HiddenStateDim { expected, got } if expected == dim && got == dim + 1 => {}
            other => panic!("unexpected {other:?}"),
        }
        assert_preflight_frozen(&net);
    }

    #[test]
    fn preflight_rejects_rank3_without_stepping() {
        let (dim, channels) = tiny_dims();
        let mut net = scripted_net(Tensor::from_vec(vec![0.0; 8], &[2, 2, 2]), dim, channels);
        match net.forward(&[1, 2], None).unwrap_err() {
            HybridError::HiddenStateRank { got: 3 } => {}
            other => panic!("unexpected {other:?}"),
        }
        assert_preflight_frozen(&net);
    }

    #[test]
    fn preflight_rejects_wrong_seq_len_without_stepping() {
        let (dim, channels) = tiny_dims();
        let mut net = scripted_net(valid_rank2(4, dim), dim, channels);
        match net.forward(&[1, 2], None).unwrap_err() {
            HybridError::HiddenStateSeqLen {
                expected: 2,
                got: 4,
            } => {}
            other => panic!("unexpected {other:?}"),
        }
        assert_preflight_frozen(&net);
    }

    #[test]
    fn preflight_rejects_wrong_hidden_dim_without_stepping() {
        let (dim, channels) = tiny_dims();
        let mut net = scripted_net(valid_rank2(3, dim + 8), dim, channels);
        match net.forward(&[1, 2, 3], None).unwrap_err() {
            HybridError::HiddenStateDim { expected, got } if expected == dim && got == dim + 8 => {}
            other => panic!("unexpected {other:?}"),
        }
        assert_preflight_frozen(&net);
    }

    #[test]
    fn preflight_rejects_empty_backing_data_without_stepping() {
        let (dim, channels) = tiny_dims();
        let mut net = scripted_net(tensor_parts(vec![], vec![2, dim]), dim, channels);
        match net.forward(&[1, 2], None).unwrap_err() {
            HybridError::HiddenStateDataLen { expected, got: 0 } if expected == 2 * dim => {}
            other => panic!("unexpected {other:?}"),
        }
        assert_preflight_frozen(&net);
    }

    #[test]
    fn preflight_rejects_inconsistent_storage_without_stepping() {
        let (dim, channels) = tiny_dims();
        let mut net = scripted_net(tensor_parts(vec![0.0; 3], vec![2, dim]), dim, channels);
        match net.forward(&[1, 2], None).unwrap_err() {
            HybridError::HiddenStateDataLen { expected, got: 3 } if expected == 2 * dim => {}
            other => panic!("unexpected {other:?}"),
        }
        assert_preflight_frozen(&net);
    }

    #[test]
    fn preflight_rejects_nan_hidden_without_stepping() {
        let (dim, channels) = tiny_dims();
        let mut hidden = valid_rank2(2, dim);
        hidden.data_mut()[4] = f32::NAN;
        let mut net = scripted_net(hidden, dim, channels);
        match net.forward(&[1, 2], None).unwrap_err() {
            HybridError::NonFinite {
                stage: ForwardValueStage::HiddenState,
                index: 4,
            } => {}
            other => panic!("unexpected {other:?}"),
        }
        assert_preflight_frozen(&net);
    }

    #[test]
    fn preflight_rejects_pos_inf_hidden_without_stepping() {
        let (dim, channels) = tiny_dims();
        let mut hidden = valid_rank2(2, dim);
        hidden.data_mut()[0] = f32::INFINITY;
        let mut net = scripted_net(hidden, dim, channels);
        match net.forward(&[1, 2], None).unwrap_err() {
            HybridError::NonFinite {
                stage: ForwardValueStage::HiddenState,
                index: 0,
            } => {}
            other => panic!("unexpected {other:?}"),
        }
        assert_preflight_frozen(&net);
    }

    #[test]
    fn preflight_rejects_neg_inf_hidden_without_stepping() {
        let (dim, channels) = tiny_dims();
        let mut hidden = valid_rank2(2, dim);
        hidden.data_mut()[1] = f32::NEG_INFINITY;
        let mut net = scripted_net(hidden, dim, channels);
        match net.forward(&[1, 2], None).unwrap_err() {
            HybridError::NonFinite {
                stage: ForwardValueStage::HiddenState,
                index: 1,
            } => {}
            other => panic!("unexpected {other:?}"),
        }
        assert_preflight_frozen(&net);
    }

    #[test]
    fn preflight_rejects_overflow_inf_embedding_without_stepping() {
        let (dim, channels) = tiny_dims();
        // Two finite f32::MAX rows sum to +Inf during mean-pool.
        let hidden = Tensor::from_vec(vec![f32::MAX; 2 * dim], &[2, dim]);
        let mut net = scripted_net(hidden, dim, channels);
        match net.forward(&[1, 2], None).unwrap_err() {
            HybridError::NonFinite {
                stage: ForwardValueStage::Embedding,
                index: 0,
            } => {}
            other => panic!("unexpected {other:?}"),
        }
        assert_preflight_frozen(&net);
    }

    #[test]
    fn preflight_rejects_zero_snn_channels_without_stepping() {
        let (dim, _) = tiny_dims();
        let mut net = scripted_net(valid_rank2(2, dim), dim, 0);
        match net.forward(&[1, 2], None).unwrap_err() {
            HybridError::ZeroSnnChannels => {}
            other => panic!("unexpected {other:?}"),
        }
        assert_preflight_frozen(&net);
    }

    #[test]
    fn preflight_rejects_zero_transformer_dim_without_stepping() {
        let (_, channels) = tiny_dims();
        let hidden = tensor_parts(vec![], vec![0]);
        let mut net = scripted_net(hidden, 0, channels);
        match net.forward(&[1, 2], None).unwrap_err() {
            HybridError::HiddenStateDim {
                expected: 1,
                got: 0,
            } => {}
            other => panic!("unexpected {other:?}"),
        }
        assert_preflight_frozen(&net);
    }

    #[test]
    fn valid_rank2_preserves_pooled_embedding_and_steps_once() {
        let (dim, channels) = tiny_dims();
        let seq = 4;
        let hidden = valid_rank2(seq, dim);
        let mut net = scripted_net(hidden, dim, channels);
        let out = net.forward(&[1, 2, 3, 4], None).expect("valid rank-2");
        assert_eq!(out.embedding.len(), dim);
        // Mean-pool of data[t * dim + i] = 0.01 * (t * dim + i) over t in 0..seq:
        // embedding[i] = 0.01 * (i + dim * (seq - 1) / 2)
        let expected_0 = 0.01 * dim as f32 * (seq - 1) as f32 / 2.0;
        assert!((out.embedding[0] - expected_0).abs() < 1e-5);
        assert_eq!(out.stimuli.len(), channels);
        for v in &out.stimuli {
            assert!(v.is_finite() && v.abs() <= 1.0);
        }
        assert_eq!(net.snn.step_calls, 1);
        assert_eq!(net.snn.state, SPY_STATE + 1);
        assert_eq!(net.global_step(), 1);
        assert_eq!(out.global_step, 1);
    }

    #[test]
    fn valid_rank1_pooled_vector_steps_once() {
        let (dim, channels) = tiny_dims();
        let data: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
        let mut net = scripted_net(Tensor::from_vec(data.clone(), &[dim]), dim, channels);
        let out = net.forward(&[1, 2, 3], None).expect("valid rank-1");
        assert_eq!(out.embedding, data);
        assert_eq!(out.stimuli.len(), channels);
        assert_eq!(net.snn.step_calls, 1);
        assert_eq!(net.global_step(), 1);
    }
}
