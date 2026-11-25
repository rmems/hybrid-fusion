// SPDX-License-Identifier: MIT OR Apache-2.0

pub mod adapter;
#[cfg(feature = "axon-encoder")]
pub mod axon_adapter;
#[cfg(feature = "backends")]
pub mod backends;
pub mod capabilities;
pub mod encoding;
pub mod error;
pub mod hybrid;
#[cfg(feature = "neuromod")]
pub mod neuromod_adapter;
pub mod plan;
pub mod projector;
pub mod reverse;
pub mod routing;
mod serde_finite;
pub mod spiking_router;
pub mod tensor;
pub mod topology;
pub mod track;
pub mod traits;
pub mod types;
mod validate;

pub use adapter::{
    DomainAdapter, MAX_ADAPTER_WIDTH, PoolResizeTanhAdapter, STIMULUS_BOUND,
    decode_activity_checked, encode_stimuli_checked, validate_decode_input, validate_embedding,
    validate_encode_input, validate_stimuli, validate_stimulus_range,
};
pub use capabilities::{
    BackendCapabilities, BackendId, CapabilityNegotiation, DeviceHint, FallbackPolicy,
    NegotiationOutcome, NegotiationReport, RequiredFeature, StageRequirement,
};
pub use encoding::{
    RateEncoder, SpikeCountDecoder, SpikeDecoder, SpikeDecoding, SpikeDecodingKind, SpikeEncoder,
    SpikeEncoding, SpikeEncodingKind,
};
pub use error::{ForwardValueStage, HybridError, Result};
pub use hybrid::HybridNetwork;
pub use plan::{
    DimSpec, Edge, EdgeKind, ExecutionDomain, HybridExecutionPlan, PlanError, PortSpec, Stage,
    StageGraph, StageId, StageKind,
};
pub use projector::{project_spike_activity, spike_activity_features};
pub use reverse::{ReverseHybridPath, WEIGHT_SUM_TOLERANCE};
pub use routing::{
    MAX_REASONABLE_EXPERTS, route_synthetic, routing_entropy, softmax, synthetic_gate_scores,
    top_k_indices,
};
pub use spiking_router::{DEFAULT_SPIKE_BOOST, SpikingExpertRouter, SpikingRouteOutput};
pub use tensor::Tensor;
pub use topology::{AnnTopology, HybridTopology, SnnTopology, TopologySource};
pub use track::ResultTrack;
pub use traits::{
    ExpertRouteOutput, ExpertRouter, NeuroModulators, SpikeActivity, SpikingNetwork, Transformer,
};
#[allow(deprecated)]
pub use traits::{GgufLayout, GgufLoader, SafetensorsLayout, SafetensorsLoader};
pub use types::{
    Dtype, HybridConfig, HybridOutput, ProjectionMode, TensorManifestEntry, TensorRole,
    TransformerConfig,
};

#[cfg(feature = "backends")]
pub use backends::simple_snn::SimpleSnn;
#[cfg(feature = "backends")]
pub use backends::simple_transformer::SimpleTransformer;
#[cfg(feature = "backends")]
pub use backends::stub_router::StubExpertRouter;
#[cfg(feature = "backends")]
pub use backends::synthetic_router::SyntheticExpertRouter;

/// Real SNN backend adapter over `neuromod` 0.7, enabled by the optional
/// `neuromod` feature. No `neuromod` type crosses this boundary.
#[cfg(feature = "neuromod")]
pub use neuromod_adapter::NeuromodSnn;

/// `DomainAdapter` backed by `axon-encoder` sensory encoders, enabled by the
/// optional `axon-encoder` feature. Encoding is owned by that crate; SNN→ANN
/// decoding stays in-tree via `PoolResizeTanhAdapter`.
#[cfg(feature = "axon-encoder")]
pub use axon_adapter::AxonEncoderAdapter;
