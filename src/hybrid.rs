// SPDX-License-Identifier: MIT OR Apache-2.0

//! ANN → SNN forward-path host: transformer hidden states → bounded SNN stimuli.

use crate::error::{HybridError, Result};
use crate::projector;
use crate::telemetry;
use crate::tensor::Tensor;
use crate::traits::{NeuroModulators, SpikingNetwork, Transformer};
use crate::types::{HybridConfig, HybridOutput};

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
        // Caller-side validation errors are returned without Sentry capture so
        // routine bad requests do not flood the error stream / quota.
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

        let hidden = self.transformer.hidden_states(token_ids);
        if hidden.ndim() == 2 && hidden.shape()[1] != self.transformer.dim() {
            return Err(HybridError::InvalidConfig(format!(
                "hidden state dim={} does not match transformer.dim()={}",
                hidden.shape()[1],
                self.transformer.dim(),
            )));
        }
        let embedding = pool_embedding(&hidden, self.transformer.dim());
        let snn_width = self.snn.num_channels();
        let stimuli = projector::embed_to_stimuli_with_width(&hidden, snn_width);

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
}
