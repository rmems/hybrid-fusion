// SPDX-License-Identifier: MIT OR Apache-2.0

//! Minimal deterministic transformer backend.
//!
//! `SimpleTransformer` uses a learned embedding lookup table and a single
//! linear projection to produce hidden states. It is useful for testing the
//! hybrid forward pass without a real LLM backend.

use crate::error::Result;
use crate::tensor::Tensor;
use crate::traits::Transformer;
use crate::types::TransformerConfig;

/// A lightweight transformer that produces deterministic hidden states via
/// embedding lookup + linear projection.
pub struct SimpleTransformer {
    vocab_size: usize,
    dim: usize,
    max_seq_len: usize,
    /// Embedding table flattened: vocab_size * dim (row-major).
    embeddings: Vec<f32>,
    /// Projection matrix dim * dim (row-major).
    projection: Vec<f32>,
}

impl SimpleTransformer {
    /// Build from a [`TransformerConfig`]. Weights are initialised with a
    /// deterministic pseudo-random sequence so behaviour is reproducible.
    pub fn new(config: TransformerConfig) -> Self {
        let vocab_size = config.vocab_size;
        let dim = config.dim;
        // Zero-sized configs later panic (modulo-by-zero in embed_token) or
        // violate Tensor::from_vec's non-zero extent invariant — reject early.
        assert!(
            vocab_size > 0,
            "SimpleTransformer::new: vocab_size must be > 0, got {vocab_size}"
        );
        assert!(
            dim > 0,
            "SimpleTransformer::new: dim must be > 0, got {dim}"
        );

        let mut lcg = Lcg::new(
            (vocab_size as u64).wrapping_mul(2654435761) ^ (dim as u64).wrapping_mul(2246822519),
        );

        // next_f32() already returns [-1, 1); scale by 0.01 → symmetric [-0.01, 0.01).
        let embeddings = (0..vocab_size * dim)
            .map(|_| lcg.next_f32() * 0.01)
            .collect();
        let projection = (0..dim * dim).map(|_| lcg.next_f32() * 0.01).collect();

        Self {
            vocab_size,
            dim,
            max_seq_len: config.max_seq_len,
            embeddings,
            projection,
        }
    }

    fn embed_token(&self, token_id: u32) -> &[f32] {
        let idx = (token_id as usize % self.vocab_size) * self.dim;
        &self.embeddings[idx..idx + self.dim]
    }

    fn project(&self, input: &[f32]) -> Vec<f32> {
        let dim = self.dim;
        let mut out = vec![0.0f32; dim];
        for (j, o) in out.iter_mut().enumerate() {
            let row = &self.projection[j * dim..(j + 1) * dim];
            *o = row.iter().zip(input).map(|(w, x)| w * x).sum::<f32>();
        }
        out
    }
}

impl Transformer for SimpleTransformer {
    fn hidden_states(&self, token_ids: &[u32]) -> Result<Tensor> {
        let seq = token_ids.len();
        let dim = self.dim;
        let mut data = Vec::with_capacity(seq * dim);
        for &tid in token_ids {
            let emb = self.embed_token(tid);
            let proj = self.project(emb);
            data.extend_from_slice(&proj);
        }
        Tensor::from_vec(data, &[seq, dim])
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn param_count(&self) -> usize {
        self.embeddings.len() + self.projection.len()
    }
}

/// Minimal deterministic 64-bit LCG for weight initialisation.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32 & 0x007F_FFFF;
        (bits as f32 / (1 << 23) as f32) * 2.0 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> TransformerConfig {
        TransformerConfig {
            vocab_size: 8,
            dim: 4,
            num_heads: 1,
            num_layers: 1,
            ff_dim: 8,
            max_seq_len: 16,
        }
    }

    #[test]
    fn hidden_states_shape() {
        let t = SimpleTransformer::new(tiny_config());
        let hs = t.hidden_states(&[0, 1, 2]).unwrap();
        assert_eq!(hs.shape(), &[3, 4]);
    }

    #[test]
    fn determinism() {
        let t1 = SimpleTransformer::new(tiny_config());
        let t2 = SimpleTransformer::new(tiny_config());
        assert_eq!(
            t1.hidden_states(&[3, 5]).unwrap().data(),
            t2.hidden_states(&[3, 5]).unwrap().data()
        );
    }

    #[test]
    fn weight_init_is_symmetric_around_zero() {
        let t = SimpleTransformer::new(tiny_config());
        // With the corrected scaling, a non-trivial fraction of embeddings
        // should be positive (the previous bias pushed almost everything < 0).
        let positives = t
            .hidden_states(&[0, 1, 2, 3, 4, 5, 6, 7])
            .unwrap()
            .data()
            .iter()
            .filter(|v| **v > 0.0)
            .count();
        assert!(
            positives > 0,
            "expected some positive hidden values after unbiased init"
        );
    }

    #[test]
    fn param_count_matches_allocations() {
        let t = SimpleTransformer::new(tiny_config());
        assert_eq!(t.param_count(), 8 * 4 + 4 * 4);
    }

    #[test]
    fn out_of_vocab_wraps() {
        let t = SimpleTransformer::new(tiny_config());
        let hs_wrap = t.hidden_states(&[100]).unwrap();
        let hs_norm = t.hidden_states(&[4]).unwrap();
        assert_eq!(hs_wrap.data(), hs_norm.data());
    }

    #[test]
    fn rejects_zero_vocab_size() {
        let mut cfg = tiny_config();
        cfg.vocab_size = 0;
        let result = std::panic::catch_unwind(|| SimpleTransformer::new(cfg));
        assert!(result.is_err(), "vocab_size == 0 must be rejected");
    }

    #[test]
    fn rejects_zero_dim() {
        let mut cfg = tiny_config();
        cfg.dim = 0;
        let result = std::panic::catch_unwind(|| SimpleTransformer::new(cfg));
        assert!(result.is_err(), "dim == 0 must be rejected");
    }
}
