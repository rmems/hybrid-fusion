// SPDX-License-Identifier: MIT OR Apache-2.0

use thiserror::Error;

#[derive(Debug, Error)]
pub enum HybridError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Configured value disagrees with a backend-reported capability, or a
    /// dimension/capacity is zero.
    ///
    /// `field` names the disagreeing key:
    /// [`FIELD_TRANSFORMER_DIM`](Self::FIELD_TRANSFORMER_DIM),
    /// [`FIELD_TRANSFORMER_MAX_SEQ_LEN`](Self::FIELD_TRANSFORMER_MAX_SEQ_LEN), or
    /// [`FIELD_SNN_INPUT_CHANNELS`](Self::FIELD_SNN_INPUT_CHANNELS).
    /// `configured` is the [`crate::HybridConfig`] value; `backend` is the
    /// matching [`crate::Transformer`] / [`crate::SpikingNetwork`] report.
    #[error("configuration mismatch for {field}: configured {configured}, backend {backend}")]
    ConfigMismatch {
        field: &'static str,
        configured: usize,
        backend: usize,
    },

    #[error("model load failed for '{path}': {reason}")]
    ModelLoad { path: String, reason: String },

    #[error("unsupported model format: {0}")]
    UnsupportedFormat(String),

    #[error("missing tensor '{name}' in model '{path}'")]
    MissingTensor { name: String, path: String },

    #[error("GGUF parse failed: {0}")]
    GgufParse(String),

    #[error("input length mismatch: expected {expected}, got {got}")]
    InputLengthMismatch { expected: usize, got: usize },

    #[error("SNN step failed: {0}")]
    SnnStep(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl HybridError {
    /// `ConfigMismatch.field` for transformer hidden size (`"transformer.dim"`).
    pub const FIELD_TRANSFORMER_DIM: &'static str = "transformer.dim";
    /// `ConfigMismatch.field` for maximum sequence length (`"transformer.max_seq_len"`).
    pub const FIELD_TRANSFORMER_MAX_SEQ_LEN: &'static str = "transformer.max_seq_len";
    /// `ConfigMismatch.field` for SNN input width (`"snn_input_channels"`).
    pub const FIELD_SNN_INPUT_CHANNELS: &'static str = "snn_input_channels";

    pub(crate) fn config_mismatch(field: &'static str, configured: usize, backend: usize) -> Self {
        Self::ConfigMismatch {
            field,
            configured,
            backend,
        }
    }
}

pub type Result<T> = std::result::Result<T, HybridError>;
