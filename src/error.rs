// SPDX-License-Identifier: MIT OR Apache-2.0

use thiserror::Error;

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

    #[error("SNN step failed: {0}")]
    SnnStep(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, HybridError>;
