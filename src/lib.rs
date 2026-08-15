// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "backends")]
pub mod backends;
pub mod error;
pub mod hybrid;
pub mod projector;
pub mod reverse;
pub mod routing;
pub mod telemetry;
pub mod tensor;
pub mod traits;
pub mod types;

pub use error::{HybridError, Result};
pub use hybrid::HybridNetwork;
pub use projector::{project_spike_activity, spike_activity_features};
pub use reverse::ReverseHybridPath;
pub use routing::{
    MAX_REASONABLE_EXPERTS, route_synthetic, routing_entropy, softmax, synthetic_gate_scores,
    top_k_indices,
};
pub use tensor::Tensor;
pub use traits::{
    ExpertRouteOutput, ExpertRouter, GgufLayout, GgufLoader, NeuroModulators, SpikeActivity,
    SpikingNetwork, Transformer,
};
pub use types::{HybridConfig, HybridOutput, ProjectionMode, TransformerConfig};

/// Re-export of the `sentry` crate when the `sentry` feature is enabled.
///
/// Downstream crates should initialise Sentry through this re-export so the
/// same crate version (and global hub) is shared with hybrid-fusion's internal
/// error capture:
///
/// ```ignore
/// let _guard = hybrid_fusion::sentry::init(hybrid_fusion::sentry::ClientOptions::default());
/// ```
#[cfg(feature = "sentry")]
pub use sentry;

#[cfg(feature = "backends")]
pub use backends::simple_snn::SimpleSnn;
#[cfg(feature = "backends")]
pub use backends::simple_transformer::SimpleTransformer;
#[cfg(feature = "backends")]
pub use backends::stub_router::StubExpertRouter;
#[cfg(feature = "backends")]
pub use backends::synthetic_router::SyntheticExpertRouter;
