// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use thiserror::Error;

/// Stage of the ANN→SNN forward path where a non-finite value was found.
///
/// Used by [`HybridError::NonFinite`] so backend-contract diagnostics can name
/// the buffer (hidden state, pooled embedding, or SNN stimuli) without treating
/// the failure as a Sentry runtime error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardValueStage {
    /// Transformer `hidden_states` backing data.
    HiddenState,
    /// Mean-pooled embedding produced for [`crate::HybridOutput::embedding`].
    Embedding,
    /// Projector output about to be passed to [`crate::SpikingNetwork::step`].
    Stimuli,
}

impl fmt::Display for ForwardValueStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HiddenState => write!(f, "hidden-state"),
            Self::Embedding => write!(f, "embedding"),
            Self::Stimuli => write!(f, "stimuli"),
        }
    }
}

#[derive(Debug, Error)]
pub enum HybridError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("model load failed for '{path}': {reason}")]
    ModelLoad { path: String, reason: String },

    #[error("unsupported model format: {0}")]
    UnsupportedFormat(String),

    #[error("missing tensor '{name}' in model '{path}'")]
    MissingTensor { name: String, path: String },

    #[error("GGUF parse failed: {0}")]
    GgufParse(String),

    #[error("Safetensors parse failed: {0}")]
    SafetensorsParse(String),

    #[error("input length mismatch: expected {expected}, got {got}")]
    InputLengthMismatch { expected: usize, got: usize },

    /// Transformer hidden-state rank is not 1 (`[dim]`) or 2 (`[seq, dim]`).
    #[error("hidden-state rank {got} is unsupported (expected 1 or 2)")]
    HiddenStateRank { got: usize },

    /// Rank-2 hidden state `shape[0]` does not equal `token_ids.len()`.
    #[error("hidden-state sequence length mismatch: expected {expected}, got {got}")]
    HiddenStateSeqLen { expected: usize, got: usize },

    /// Hidden-state width does not equal [`crate::Transformer::dim`].
    #[error("hidden-state dimension mismatch: expected {expected}, got {got}")]
    HiddenStateDim { expected: usize, got: usize },

    /// Backing `data.len()` does not match the layout product (`seq * dim` or `dim`).
    #[error("hidden-state data length mismatch: expected {expected}, got {got}")]
    HiddenStateDataLen { expected: usize, got: usize },

    /// NaN or ±Inf in a forward-path buffer, before [`crate::SpikingNetwork::step`].
    #[error("non-finite value in {stage} at index {index}")]
    NonFinite {
        stage: ForwardValueStage,
        index: usize,
    },

    /// [`crate::SpikingNetwork::num_channels`] returned 0.
    #[error("SNN has zero input channels")]
    ZeroSnnChannels,

    #[error("SNN step failed: {0}")]
    SnnStep(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, HybridError>;
