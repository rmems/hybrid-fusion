// SPDX-License-Identifier: MIT OR Apache-2.0

//! Format-neutral topology input for [`HybridExecutionPlan`](crate::HybridExecutionPlan).
//!
//! Canonical planning reads four positive dimensions: ANN hidden width, ANN
//! maximum sequence length, SNN input channels, and SNN output population.
//! Checkpoint files are not opened here, and no GGUF or Safetensors type is
//! required.
//!
//! The ANN half may come from a [`TopologySource`] (a future engram-parser
//! adapter, or any other caller). The SNN half always comes from the caller
//! or from SNN configuration. Checkpoint inventory never supplies SNN
//! dimensions. Missing ANN values are errors; this module does not invent
//! defaults.

use serde::{Deserialize, Serialize};

use crate::plan::PlanError;

/// ANN dimensions canonical planning reads.
///
/// `architecture` is a descriptive label only. It does not select planning
/// behavior: two values that differ only in `architecture` compile to the
/// same execution plan.
///
/// These dimensions can be built by the caller or returned by a
/// [`TopologySource`]. They are not a checkpoint layout.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "AnnTopologyWire")]
pub struct AnnTopology {
    /// Hidden width. Must be `> 0`.
    pub hidden_dim: usize,
    /// Maximum sequence length. Must be `> 0`.
    pub max_seq_len: usize,
    /// Optional architecture label (for example a model-family name).
    ///
    /// Planning ignores this string. Absence means the caller supplied no
    /// label; it is not a guessed family.
    pub architecture: Option<String>,
}

#[derive(Deserialize)]
struct AnnTopologyWire {
    hidden_dim: usize,
    max_seq_len: usize,
    #[serde(default)]
    architecture: Option<String>,
}

impl TryFrom<AnnTopologyWire> for AnnTopology {
    type Error = PlanError;

    fn try_from(wire: AnnTopologyWire) -> Result<Self, Self::Error> {
        Self::new(wire.hidden_dim, wire.max_seq_len, wire.architecture)
    }
}

impl AnnTopology {
    /// Stable [`PlanError::InvalidTopology`] discriminator for hidden width.
    pub const FIELD_HIDDEN_DIM: &str = "hidden_dim";
    /// Stable [`PlanError::InvalidTopology`] discriminator for the sequence window.
    pub const FIELD_MAX_SEQ_LEN: &str = "max_seq_len";

    /// Build ANN topology.
    ///
    /// # Errors
    ///
    /// [`PlanError::InvalidTopology`] when `hidden_dim` or `max_seq_len` is `0`.
    /// `field` is [`Self::FIELD_HIDDEN_DIM`] or [`Self::FIELD_MAX_SEQ_LEN`].
    pub fn new(
        hidden_dim: usize,
        max_seq_len: usize,
        architecture: Option<String>,
    ) -> Result<Self, PlanError> {
        require_positive(Self::FIELD_HIDDEN_DIM, hidden_dim)?;
        require_positive(Self::FIELD_MAX_SEQ_LEN, max_seq_len)?;
        Ok(Self {
            hidden_dim,
            max_seq_len,
            architecture,
        })
    }
}

/// SNN dimensions canonical planning reads.
///
/// Input channels and output population are independent axes. Neither is
/// derived from the other, and neither is read from checkpoint inventory.
/// The caller or the SNN configuration supplies both.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "SnnTopologyWire")]
pub struct SnnTopology {
    /// Stimulus width passed to the spiking stage. Must be `> 0`.
    /// Independent of [`output_population`](Self::output_population).
    pub input_channels: usize,
    /// Fired-neuron population. Must be `> 0`.
    /// Independent of [`input_channels`](Self::input_channels).
    pub output_population: usize,
}

#[derive(Deserialize)]
struct SnnTopologyWire {
    input_channels: usize,
    output_population: usize,
}

impl TryFrom<SnnTopologyWire> for SnnTopology {
    type Error = PlanError;

    fn try_from(wire: SnnTopologyWire) -> Result<Self, Self::Error> {
        Self::new(wire.input_channels, wire.output_population)
    }
}

impl SnnTopology {
    /// Stable [`PlanError::InvalidTopology`] discriminator for stimulus width.
    pub const FIELD_INPUT_CHANNELS: &str = "input_channels";
    /// Stable [`PlanError::InvalidTopology`] discriminator for output population.
    pub const FIELD_OUTPUT_POPULATION: &str = "output_population";

    /// Build SNN topology.
    ///
    /// # Errors
    ///
    /// [`PlanError::InvalidTopology`] when `input_channels` or
    /// `output_population` is `0`. `field` is [`Self::FIELD_INPUT_CHANNELS`] or
    /// [`Self::FIELD_OUTPUT_POPULATION`].
    pub fn new(input_channels: usize, output_population: usize) -> Result<Self, PlanError> {
        require_positive(Self::FIELD_INPUT_CHANNELS, input_channels)?;
        require_positive(Self::FIELD_OUTPUT_POPULATION, output_population)?;
        Ok(Self {
            input_channels,
            output_population,
        })
    }
}

/// ANN and SNN dimensions consumed by canonical planning.
///
/// The ANN part can come from a [`TopologySource`]. The SNN part always comes
/// from the caller or SNN configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "HybridTopologyWire")]
pub struct HybridTopology {
    /// Transformer-side dimensions.
    pub ann: AnnTopology,
    /// Spiking-side dimensions. Not derived from [`ann`](Self::ann).
    pub snn: SnnTopology,
}

#[derive(Deserialize)]
struct HybridTopologyWire {
    ann: AnnTopology,
    snn: SnnTopology,
}

impl TryFrom<HybridTopologyWire> for HybridTopology {
    type Error = PlanError;

    fn try_from(wire: HybridTopologyWire) -> Result<Self, Self::Error> {
        Self::new(wire.ann, wire.snn)
    }
}

impl HybridTopology {
    /// Combine already-described ANN and SNN halves.
    ///
    /// Re-checks every dimension so a value mutated after construction is
    /// still rejected.
    ///
    /// # Errors
    ///
    /// [`PlanError::InvalidTopology`] when any dimension is `0`.
    pub fn new(ann: AnnTopology, snn: SnnTopology) -> Result<Self, PlanError> {
        require_positive(AnnTopology::FIELD_HIDDEN_DIM, ann.hidden_dim)?;
        require_positive(AnnTopology::FIELD_MAX_SEQ_LEN, ann.max_seq_len)?;
        require_positive(SnnTopology::FIELD_INPUT_CHANNELS, snn.input_channels)?;
        require_positive(SnnTopology::FIELD_OUTPUT_POPULATION, snn.output_population)?;
        Ok(Self { ann, snn })
    }
}

/// Supplies the ANN half of a [`HybridTopology`].
///
/// This is the integration point for checkpoint-derived topology. Downstream
/// code implements the trait. This crate does not depend on engram-parser and
/// does not open checkpoint files. A feature-gated adapter waits until a
/// released engram-parser topology API exists and a dependency exception is
/// authorized.
///
/// # Contract
///
/// Return [`PlanError::InvalidTopology`] when hidden width or maximum sequence
/// length is missing, ambiguous, or zero. Do not substitute a default, and do
/// not derive SNN dimensions from checkpoint metadata.
pub trait TopologySource {
    /// ANN dimensions for planning.
    ///
    /// # Errors
    ///
    /// [`PlanError::InvalidTopology`] when a required ANN value is absent or
    /// not positive.
    fn ann_topology(&self) -> Result<AnnTopology, PlanError>;
}

fn require_positive(field: &'static str, value: usize) -> Result<(), PlanError> {
    if value == 0 {
        Err(PlanError::InvalidTopology {
            field,
            reason: format!("{field} must be > 0"),
        })
    } else {
        Ok(())
    }
}
