// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::{HybridError, Result};
use crate::projector;
use crate::telemetry;
use crate::tensor::Tensor;
use crate::traits::{NeuroModulators, SpikingNetwork, Transformer};
use crate::types::{HybridConfig, HybridOutput};

pub struct HybridNetwork<T: Transformer, S: SpikingNetwork> {
    pub transformer: T,
    pub snn: S,
    config: HybridConfig,
    global_step: u64,
}

impl<T: Transformer, S: SpikingNetwork> HybridNetwork<T, S> {
    pub fn new(transformer: T, snn: S, config: HybridConfig) -> Self {
        Self {
            transformer,
            snn,
            config,
            global_step: 0,
        }
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
            // Reverse-path MoE fields stay unset on the ANN→SNN forward pass
            // (see ExpertRouter / SpikeActivity; full reverse orchestration is later).
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
        let mut net = HybridNetwork::new(t, s, cfg);
        let out = net.forward(&[0, 1, 2], None).unwrap();
        assert_eq!(out.stimuli.len(), 7);
        assert_eq!(out.embedding.len(), 128);
    }
}
