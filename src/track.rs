// SPDX-License-Identifier: MIT OR Apache-2.0

//! Result-track labels (issue #62, Linear RM-2207; shape fixed by
//! [ADR 0001 §5.1](https://github.com/rmems/hybrid-fusion/blob/main/docs/adr/0001-snn-step-contract.md)).
//!
//! ANN, native-SNN, and hybrid results are kept in separate tracks so a number
//! from one is never silently compared with a number from another. Every
//! result record the hybrid hosts emit carries a required [`ResultTrack`] field,
//! set by the host (never by a backend):
//!
//! | Record | Emitted by | Track |
//! |---|---|---|
//! | [`HybridOutput`](crate::HybridOutput) | [`HybridNetwork::forward`](crate::HybridNetwork::forward) | [`Hybrid`](ResultTrack::Hybrid) |
//! | [`HybridOutput`](crate::HybridOutput) | [`ReverseHybridPath::forward_activity`](crate::ReverseHybridPath::forward_activity) | [`Hybrid`](ResultTrack::Hybrid) |
//! | [`SpikingRouteOutput`](crate::SpikingRouteOutput) | [`SpikingExpertRouter::forward`](crate::SpikingExpertRouter::forward) / [`forward_frozen`](crate::SpikingExpertRouter::forward_frozen) | [`Hybrid`](ResultTrack::Hybrid) |
//!
//! These hosts cross the ANN/SNN boundary, so their records are `Hybrid`.
//! A plain [`ExpertRouter::route`](crate::ExpertRouter::route) is semantically
//! [`Ann`](ResultTrack::Ann), but its [`ExpertRouteOutput`](crate::ExpertRouteOutput)
//! currently has no track field. That is a known gap, not a labelled result.
//! The same type nested in `SpikingRouteOutput.route` is a component of the
//! enclosing hybrid record, not a separately labelled ANN result (ADR 0001
//! §5.1). [`SnnNative`](ResultTrack::SnnNative) is reserved for emitters such as
//! Spikenaut-SNN / brainstem-daemon that never call a `Transformer`.
//!
//! # Compile-time guarantee
//!
//! The field is required and the record types have no `Default`, so a record
//! cannot be built without naming a track:
//!
//! ```compile_fail,E0063
//! let _untracked = hybrid_fusion::HybridOutput {
//!     embedding: vec![0.0],
//!     stimuli: vec![0.0],
//!     fired_neurons: vec![],
//!     num_neurons: 1,
//!     global_step: 0,
//!     expert_weights: None,
//!     selected_experts: None,
//!     routing_entropy: None,
//! };
//! ```
//!
//! The same literal with the track compiles:
//!
//! ```
//! let tracked = hybrid_fusion::HybridOutput {
//!     embedding: vec![0.0],
//!     stimuli: vec![0.0],
//!     fired_neurons: vec![],
//!     num_neurons: 1,
//!     global_step: 0,
//!     expert_weights: None,
//!     selected_experts: None,
//!     routing_entropy: None,
//!     track: hybrid_fusion::ResultTrack::Hybrid,
//! };
//! assert_eq!(tracked.track, hybrid_fusion::ResultTrack::Hybrid);
//! ```
//!
//! Deserialization is equally strict: a record without a `track` key is
//! rejected (`` missing field `track` ``), and an unknown wire value is rejected
//! rather than mapped to a default.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::{HybridError, Result};

/// Which result track a record belongs to.
///
/// Wire form (serde, `Display`, [`as_str`](Self::as_str)): `ann`,
/// `snn-native`, `hybrid`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ResultTrack {
    /// Transformer / dense result only; no spiking network in the loop.
    Ann,
    /// Spiking-network result produced without a `Transformer` in the loop.
    SnnNative,
    /// Result that crosses the ANN/SNN boundary through hybrid-fusion.
    Hybrid,
}

impl ResultTrack {
    /// Wire spelling of this track (`"ann"`, `"snn-native"`, `"hybrid"`).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ann => "ann",
            Self::SnnNative => "snn-native",
            Self::Hybrid => "hybrid",
        }
    }

    /// Guard for any API that combines two results: returns the shared track,
    /// or [`HybridError::TrackMismatch`] when the tracks differ.
    ///
    /// ```
    /// use hybrid_fusion::{HybridError, ResultTrack};
    ///
    /// assert_eq!(
    ///     ResultTrack::Hybrid.require_same(ResultTrack::Hybrid).unwrap(),
    ///     ResultTrack::Hybrid
    /// );
    /// assert!(matches!(
    ///     ResultTrack::Ann.require_same(ResultTrack::Hybrid),
    ///     Err(HybridError::TrackMismatch { left: ResultTrack::Ann, right: ResultTrack::Hybrid })
    /// ));
    /// ```
    pub fn require_same(self, other: Self) -> Result<Self> {
        if self == other {
            Ok(self)
        } else {
            Err(HybridError::TrackMismatch {
                left: self,
                right: other,
            })
        }
    }
}

impl fmt::Display for ResultTrack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [(ResultTrack, &str); 3] = [
        (ResultTrack::Ann, "ann"),
        (ResultTrack::SnnNative, "snn-native"),
        (ResultTrack::Hybrid, "hybrid"),
    ];

    #[test]
    fn wire_form_round_trips_for_every_variant() {
        for (track, wire) in ALL {
            let json = serde_json::to_string(&track).unwrap();
            assert_eq!(json, format!("\"{wire}\""));
            let back: ResultTrack = serde_json::from_str(&json).unwrap();
            assert_eq!(back, track);
            assert_eq!(track.as_str(), wire);
            assert_eq!(track.to_string(), wire);
        }
    }

    #[test]
    fn unknown_or_rust_spelled_wire_values_are_rejected() {
        for bad in [
            "\"snn\"",
            "\"SnnNative\"",
            "\"snn_native\"",
            "\"Hybrid\"",
            "\"\"",
            "null",
        ] {
            assert!(
                serde_json::from_str::<ResultTrack>(bad).is_err(),
                "{bad} must not decode to a track"
            );
        }
    }

    #[test]
    fn require_same_accepts_equal_and_rejects_every_mismatch() {
        for (left, _) in ALL {
            for (right, _) in ALL {
                let got = left.require_same(right);
                if left == right {
                    assert_eq!(got.unwrap(), left);
                } else {
                    match got {
                        Err(HybridError::TrackMismatch { left: l, right: r }) => {
                            assert_eq!((l, r), (left, right));
                        }
                        other => panic!("expected TrackMismatch, got {other:?}"),
                    }
                }
            }
        }
    }

    #[test]
    fn track_mismatch_error_names_both_wire_values() {
        let err = ResultTrack::SnnNative
            .require_same(ResultTrack::Ann)
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "result track mismatch: snn-native vs ann (cross-track results cannot be combined)"
        );
    }
}
