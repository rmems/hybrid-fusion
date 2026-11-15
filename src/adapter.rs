// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`DomainAdapter`]: the ANN↔SNN representation boundary (RM-1811).
//!
//! A domain adapter owns the two representation changes that cross the
//! boundary between a transformer and a spiking network:
//!
//! - **ANN → SNN** ([`DomainAdapter::encode_stimuli`]): transformer hidden
//!   states → one bounded stimulus vector, one value per SNN input channel.
//! - **SNN → ANN** ([`DomainAdapter::decode_activity`]): recorded
//!   [`SpikeActivity`] → one dense, finite embedding.
//!
//! [`PoolResizeTanhAdapter`] is the reference implementation: the existing
//! mean-pool → resize → `tanh` forward path and the pure
//! [`ProjectionMode`] reverse path, behind one contract. Rate coding, temporal
//! coding, learned encoders / readouts, spike-count decoding and
//! membrane-state decoding are further implementations of the same trait;
//! none of them is hard-coded into an orchestration host.
//!
//! # Scope
//!
//! This module defines and enforces the contract. [`crate::HybridNetwork`] and
//! [`crate::ReverseHybridPath`] still call the projector directly and are not
//! routed through an adapter yet; the reference adapter is bit-for-bit
//! equivalent to that path (tested against
//! [`crate::projector::embed_to_stimuli_with_width`] and
//! [`crate::project_spike_activity`]).
//!
//! # Enforcement
//!
//! Contract violations are returned as [`HybridError`] values, never panics:
//!
//! - **Inputs** are checked by [`validate_encode_input`] /
//!   [`validate_decode_input`].
//! - **Outputs** are checked by [`validate_stimuli`] / [`validate_embedding`].
//! - Hosts call the provided [`encode_stimuli_checked`] /
//!   [`decode_activity_checked`] free functions, which run input validation,
//!   the implementation hook, then output validation. They are functions, not
//!   trait methods, so no implementation can override or bypass them.
//! - [`conformance`] is a reusable test helper every implementation must pass.

pub mod conformance;

use crate::encoding::{SpikeDecoding, SpikeDecodingKind, SpikeEncoding, SpikeEncodingKind};
use crate::error::{ForwardValueStage, HybridError, Result};
use crate::projector;
use crate::tensor::Tensor;
use crate::traits::SpikeActivity;
use crate::types::ProjectionMode;
use crate::validate;

/// Largest width (`snn_width`, `num_neurons`, `embed_dim`) an adapter accepts.
///
/// `1 << 24` `f32` values is 64 MiB, the same order as the `neuromod` adapter's
/// matrix budget. The cap exists so an absurd, backend- or config-supplied
/// width is a returned [`HybridError::InvalidConfig`] rather than a
/// capacity-overflow panic or allocator abort inside an implementation.
pub const MAX_ADAPTER_WIDTH: usize = 1 << 24;

/// Stimuli handed to [`crate::SpikingNetwork::step`] lie in `[-STIMULUS_BOUND,
/// STIMULUS_BOUND]` (crate invariant; `tanh` enforces it in the reference
/// adapter).
pub const STIMULUS_BOUND: f32 = 1.0;

/// ANN↔SNN representation boundary. See the [module docs](self) for scope.
///
/// # Contract
///
/// ## Dtype
///
/// Every buffer is `f32`. [`Tensor`] cannot hold any other dtype, and
/// [`SpikeActivity`] potentials are `f32`. There is no implicit conversion.
///
/// ## Encode: ANN → SNN
///
/// [`encode_stimuli`](Self::encode_stimuli)`(hidden, snn_width)`
///
/// *Input shape.* `hidden` is a row-major tensor of rank 1 (`[dim]`) or rank 2
/// (`[seq, dim]`); every extent is `> 0`; `data().len()` equals the product of
/// the extents; every value is finite. `snn_width` is the SNN input channel
/// count, in `1..=`[`MAX_ADAPTER_WIDTH`]. These are the same rules the forward
/// host applies to a transformer's hidden state.
///
/// *Output shape.* Exactly `snn_width` values. Every value is finite and in
/// `[-1, 1]` ([`STIMULUS_BOUND`]). If [`encoding`](Self::encoding) declares a
/// `range`, every value also lies inside it.
///
/// *Errors.* [`validate_encode_input`] defines the exact variant for each input
/// violation. An output that breaks the output shape is reported by
/// [`validate_stimuli`]. Failures are returned, never panicked.
///
/// ## Decode: SNN → ANN
///
/// [`decode_activity`](Self::decode_activity)`(activity, num_neurons, embed_dim)`
///
/// *Input shape.* `num_neurons` is the SNN **output population** (not the
/// input width) in `1..=`[`MAX_ADAPTER_WIDTH`]. `activity` obeys the
/// [`crate::spike_activity_features`] input contract:
/// `potentials.len() == num_neurons` and finite; each `spike_train` step is a
/// *set* of in-range neuron indices (no repeats); `iz_potentials` is empty or
/// `num_neurons` long and finite. `embed_dim` is in `1..=`[`MAX_ADAPTER_WIDTH`].
///
/// *Output shape.* Exactly `embed_dim` values, every one finite. A bound
/// tighter than "finite" is a property of the implementation (the reference
/// adapter emits `[-1, 1]`), not of the trait.
///
/// *Errors.* [`validate_decode_input`] / [`validate_embedding`], as above.
///
/// ## Determinism
///
/// An adapter is a pure function of `(adapter state, arguments)`: no clock,
/// no global or thread-local state, no unseeded randomness. Two adapters built
/// the same way and fed the same call sequence produce **bitwise-identical**
/// outputs. A stochastic encoder must carry its seeded generator as adapter
/// state, and [`reset`](Self::reset) must restore that seed.
///
/// ## Statefulness
///
/// Both hooks take `&mut self` because real adapters are stateful (phase
/// accumulators, temporal-coding windows, learned-readout state). An adapter
/// that is stateless leaves [`reset`](Self::reset) at its no-op default.
///
/// - **Failure atomicity.** A call that returns `Err` leaves the adapter
///   exactly as it was. A stateful implementation computes and validates its
///   output *before* committing state. This is deliberately stricter than
///   [`crate::SpikeEncoder::encode`], whose state advances regardless.
/// - **Reset.** After [`reset`](Self::reset) the adapter behaves like a
///   freshly constructed one, for encode and decode alike.
/// - **Independence.** The adapter never mutates its arguments and keeps no
///   reference to them past the call.
///
/// ## No panics
///
/// Neither hook may panic on *any* argument, including malformed tensors
/// (a [`Tensor`] deserialized from JSON can carry a `shape` that disagrees
/// with its data) and hostile activities. Hooks defend themselves; hosts
/// additionally go through the `*_checked` functions.
///
/// # Implementing
///
/// Implement [`encode_stimuli`](Self::encode_stimuli) and
/// [`decode_activity`](Self::decode_activity); call
/// [`validate_encode_input`] / [`validate_decode_input`] first and
/// [`validate_stimuli`] / [`validate_embedding`] (plus
/// [`validate_stimulus_range`] when [`encoding`](Self::encoding) declares a
/// range) on the result so direct callers get the same guarantees as hosts. The `*_checked` entry points are
/// free functions — they cannot be overridden, so the validation boundary
/// holds for every implementation. Then prove the implementation with
/// [`conformance::assert_domain_adapter_contract`] (it takes a factory so it can
/// build fresh instances to compare against):
///
/// ```
/// use hybrid_fusion::adapter::conformance::assert_domain_adapter_contract;
/// use hybrid_fusion::{PoolResizeTanhAdapter, ProjectionMode};
///
/// assert_domain_adapter_contract(|| PoolResizeTanhAdapter::new(ProjectionMode::RateSum));
/// ```
///
/// The `Send` supertrait keeps hosts holding `Box<dyn DomainAdapter>` `Send`
/// when their other backends are — an adapter behind `Rc` or raw pointers
/// cannot be injected. `Sync` is deliberately not required: all mutation goes
/// through `&mut self`, so `Cell`-style streaming state stays legal.
pub trait DomainAdapter: Send {
    /// ANN → SNN hook. Contract: see the "Encode" section of the trait docs.
    fn encode_stimuli(&mut self, hidden: &Tensor, snn_width: usize) -> Result<Vec<f32>>;

    /// SNN → ANN hook. Contract: see the "Decode" section of the trait docs.
    fn decode_activity(
        &mut self,
        activity: &SpikeActivity,
        num_neurons: usize,
        embed_dim: usize,
    ) -> Result<Vec<f32>>;

    /// Restore the adapter to its freshly constructed state (encode and
    /// decode). Stateless adapters keep this no-op default.
    fn reset(&mut self) {}

    /// Declared encoding scheme and stimulus range, for plan ports
    /// ([`crate::PortSpec::encoding`]). Default: rate, one timestep,
    /// unbounded; override it. A declared `range` must contain every encoded
    /// stimulus — enforced by [`encode_stimuli_checked`] at runtime and by
    /// [`conformance`].
    fn encoding(&self) -> SpikeEncoding {
        SpikeEncoding::default()
    }

    /// Declared decoding scheme, for plan ports ([`crate::PortSpec::decoding`]).
    /// Default: count over a one-tick window; override it. A `window` of
    /// `usize::MAX` declares "the whole recorded activity" — the honest value
    /// for a decoder that aggregates every step it is handed.
    fn decoding(&self) -> SpikeDecoding {
        SpikeDecoding::default()
    }
}

/// Host entry point for encoding: validates the input, runs the
/// [`DomainAdapter::encode_stimuli`] hook, then validates the output against
/// [`validate_stimuli`] and the adapter's declared [`SpikeEncoding::range`].
///
/// A free function, not a trait method: the validation boundary cannot be
/// overridden or bypassed by an implementation.
pub fn encode_stimuli_checked<A: DomainAdapter + ?Sized>(
    adapter: &mut A,
    hidden: &Tensor,
    snn_width: usize,
) -> Result<Vec<f32>> {
    validate_encode_input(hidden, snn_width)?;
    let stimuli = adapter.encode_stimuli(hidden, snn_width)?;
    validate_stimuli(&stimuli, snn_width)?;
    if let Some(range) = adapter.encoding().range {
        validate_stimulus_range(&stimuli, range)?;
    }
    Ok(stimuli)
}

/// Host entry point for decoding: validates the input, runs the
/// [`DomainAdapter::decode_activity`] hook, then validates the output.
///
/// A free function, not a trait method: the validation boundary cannot be
/// overridden or bypassed by an implementation.
pub fn decode_activity_checked<A: DomainAdapter + ?Sized>(
    adapter: &mut A,
    activity: &SpikeActivity,
    num_neurons: usize,
    embed_dim: usize,
) -> Result<Vec<f32>> {
    validate_decode_input(activity, num_neurons, embed_dim)?;
    let embedding = adapter.decode_activity(activity, num_neurons, embed_dim)?;
    validate_embedding(&embedding, embed_dim)?;
    Ok(embedding)
}

/// Reject an encode call that breaks the [`DomainAdapter`] input contract.
///
/// Checks run in this order, so the first violated rule names the error:
///
/// | Violation | Error |
/// |---|---|
/// | `snn_width == 0` | [`HybridError::ZeroSnnChannels`] |
/// | `snn_width > MAX_ADAPTER_WIDTH` | [`HybridError::InvalidConfig`] |
/// | rank not 1 or 2 | [`HybridError::HiddenStateRank`] |
/// | `seq == 0` | [`HybridError::HiddenStateSeqLen`] `{ expected: 1, got: 0 }` |
/// | `dim == 0` | [`HybridError::HiddenStateDim`] `{ expected: 1, got: 0 }` |
/// | `data.len() != seq * dim` (or the product overflows) | [`HybridError::HiddenStateDataLen`] |
/// | NaN / ±Inf in `data` | [`HybridError::NonFinite`] (`HiddenState`) |
///
/// Never panics, including for tensors whose `shape` disagrees with their data.
pub fn validate_encode_input(hidden: &Tensor, snn_width: usize) -> Result<()> {
    validate::validate_snn_width(snn_width)?;
    check_width_cap("snn_width", snn_width)?;
    let (seq, dim) = match *hidden.shape() {
        [dim] => (1, dim),
        [seq, dim] => (seq, dim),
        ref other => return Err(HybridError::HiddenStateRank { got: other.len() }),
    };
    if seq == 0 {
        return Err(HybridError::HiddenStateSeqLen {
            expected: 1,
            got: 0,
        });
    }
    if dim == 0 {
        return Err(HybridError::HiddenStateDim {
            expected: 1,
            got: 0,
        });
    }
    let got = hidden.data().len();
    let expected = seq
        .checked_mul(dim)
        .ok_or(HybridError::HiddenStateDataLen {
            expected: usize::MAX,
            got,
        })?;
    if got != expected {
        return Err(HybridError::HiddenStateDataLen { expected, got });
    }
    validate::validate_finite(hidden.data(), ForwardValueStage::HiddenState)
}

/// Reject stimuli that break the [`DomainAdapter`] encode output contract.
///
/// | Violation | Error |
/// |---|---|
/// | `stimuli.len() != snn_width` | [`HybridError::InputLengthMismatch`] |
/// | NaN / ±Inf | [`HybridError::NonFinite`] (`Stimuli`) |
/// | any value with magnitude above `1` | [`HybridError::InvalidConfig`] naming the index and value |
pub fn validate_stimuli(stimuli: &[f32], snn_width: usize) -> Result<()> {
    if stimuli.len() != snn_width {
        return Err(HybridError::InputLengthMismatch {
            expected: snn_width,
            got: stimuli.len(),
        });
    }
    validate::validate_finite(stimuli, ForwardValueStage::Stimuli)?;
    if let Some((index, value)) = stimuli
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| v.abs() > STIMULUS_BOUND)
    {
        return Err(HybridError::InvalidConfig(format!(
            "DomainAdapter stimulus {index} = {value} is outside [-{STIMULUS_BOUND}, {STIMULUS_BOUND}]"
        )));
    }
    Ok(())
}

/// Reject stimuli outside a declared [`SpikeEncoding::range`] `(lo, hi)`.
///
/// [`encode_stimuli_checked`] runs this after [`validate_stimuli`]. Stateful
/// implementations should call it on their output *before* committing state,
/// so a declared-range breach cannot fail the checked call after the hook has
/// already committed (failure atomicity).
///
/// | Violation | Error |
/// |---|---|
/// | any value below `lo` or above `hi` | [`HybridError::InvalidConfig`] naming the index and value |
pub fn validate_stimulus_range(stimuli: &[f32], range: (f32, f32)) -> Result<()> {
    let (lo, hi) = range;
    if let Some((index, value)) = stimuli
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| *v < lo || *v > hi)
    {
        return Err(HybridError::InvalidConfig(format!(
            "DomainAdapter stimulus {index} = {value} is outside the declared encoding range ({lo}, {hi})"
        )));
    }
    Ok(())
}

/// Reject a decode call that breaks the [`DomainAdapter`] input contract.
///
/// | Violation | Error |
/// |---|---|
/// | `embed_dim == 0` or `> MAX_ADAPTER_WIDTH` | [`HybridError::InvalidConfig`] |
/// | `num_neurons == 0` or `> MAX_ADAPTER_WIDTH` | [`HybridError::InvalidConfig`] |
/// | `potentials` / `iz_potentials` of the wrong length | [`HybridError::InputLengthMismatch`] |
/// | non-finite (iz) potential, out-of-range or repeated spike index | [`HybridError::InvalidConfig`] |
///
/// The activity rules are exactly those of [`crate::spike_activity_features`].
pub fn validate_decode_input(
    activity: &SpikeActivity,
    num_neurons: usize,
    embed_dim: usize,
) -> Result<()> {
    if embed_dim == 0 {
        return Err(HybridError::InvalidConfig(
            "DomainAdapter embed_dim must be > 0".into(),
        ));
    }
    check_width_cap("embed_dim", embed_dim)?;
    check_width_cap("num_neurons", num_neurons)?;
    projector::validate_activity(activity, num_neurons, "DomainAdapter::decode_activity")
}

/// Reject an embedding that breaks the [`DomainAdapter`] decode output
/// contract.
///
/// | Violation | Error |
/// |---|---|
/// | `embedding.len() != embed_dim` | [`HybridError::InputLengthMismatch`] |
/// | NaN / ±Inf | [`HybridError::NonFinite`] (`Embedding`) |
pub fn validate_embedding(embedding: &[f32], embed_dim: usize) -> Result<()> {
    if embedding.len() != embed_dim {
        return Err(HybridError::InputLengthMismatch {
            expected: embed_dim,
            got: embedding.len(),
        });
    }
    validate::validate_finite(embedding, ForwardValueStage::Embedding)
}

fn check_width_cap(name: &str, value: usize) -> Result<()> {
    if value > MAX_ADAPTER_WIDTH {
        return Err(HybridError::InvalidConfig(format!(
            "DomainAdapter {name} {value} exceeds MAX_ADAPTER_WIDTH ({MAX_ADAPTER_WIDTH})"
        )));
    }
    Ok(())
}

/// Reference [`DomainAdapter`]: the crate's pure, stateless projector path.
///
/// - **Encode:** mean-pool the hidden state to `[dim]`, resize to `snn_width`
///   (window-mean down, zero-pad up), then `tanh`. `tanh` runs after the
///   resize, never before.
/// - **Decode:** the configured [`ProjectionMode`] features, resized to
///   `embed_dim`, then `tanh` ([`crate::project_spike_activity`]).
///
/// Stateless, so it is trivially deterministic, failure-atomic, and
/// [`reset`](DomainAdapter::reset) is a no-op.
///
/// Because pooling and resizing average `f32` values, finite inputs of extreme
/// magnitude can overflow to `±Inf` and then `NaN` (`+Inf` and `-Inf` in one
/// resize window). The pooled and resized buffers are checked for finiteness
/// before `tanh` runs — `tanh` squashes `±Inf` to `±1` and would mask the
/// overflow — so the overflow is reported as [`HybridError::NonFinite`]
/// (`embedding` stage, matching the forward host's finite-pooled check), not
/// emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoolResizeTanhAdapter {
    mode: ProjectionMode,
}

impl PoolResizeTanhAdapter {
    /// Build the adapter with the reverse-path `mode` used by
    /// [`decode_activity`](DomainAdapter::decode_activity).
    pub const fn new(mode: ProjectionMode) -> Self {
        Self { mode }
    }

    /// Reverse-path projection mode.
    pub const fn mode(&self) -> ProjectionMode {
        self.mode
    }
}

impl DomainAdapter for PoolResizeTanhAdapter {
    fn encode_stimuli(&mut self, hidden: &Tensor, snn_width: usize) -> Result<Vec<f32>> {
        validate_encode_input(hidden, snn_width)?;
        let pooled = projector::mean_pool(hidden);
        validate::validate_finite(&pooled, ForwardValueStage::Embedding)?;
        let mut resized = projector::resize_to(&pooled, snn_width);
        validate::validate_finite(&resized, ForwardValueStage::Embedding)?;
        for v in &mut resized {
            *v = v.tanh();
        }
        validate_stimuli(&resized, snn_width)?;
        Ok(resized)
    }

    fn decode_activity(
        &mut self,
        activity: &SpikeActivity,
        num_neurons: usize,
        embed_dim: usize,
    ) -> Result<Vec<f32>> {
        validate_decode_input(activity, num_neurons, embed_dim)?;
        let embedding =
            projector::project_spike_activity(self.mode, activity, num_neurons, embed_dim)?;
        validate_embedding(&embedding, embed_dim)?;
        Ok(embedding)
    }

    fn encoding(&self) -> SpikeEncoding {
        SpikeEncoding::new(
            SpikeEncodingKind::Other("pool_resize_tanh".into()),
            1,
            Some((-STIMULUS_BOUND, STIMULUS_BOUND)),
        )
    }

    fn decoding(&self) -> SpikeDecoding {
        let (kind, window) = match self.mode {
            // Rate and histogram features aggregate every step of the recorded
            // activity, so the decode window is the whole record: `usize::MAX`
            // stands for "all provided ticks" (a fixed window would
            // misdescribe multi-step input).
            ProjectionMode::RateSum | ProjectionMode::SpikingTernary => {
                (SpikeDecodingKind::Rate, usize::MAX)
            }
            ProjectionMode::TemporalHistogram => (
                SpikeDecodingKind::Other("temporal_histogram".into()),
                usize::MAX,
            ),
            // A membrane snapshot reads the end state of the record: one tick.
            ProjectionMode::MembraneSnapshot => (SpikeDecodingKind::Membrane, 1),
        };
        SpikeDecoding::new(kind, window)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::conformance::check_domain_adapter_contract;
    use serde_json::json;

    fn malformed(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        serde_json::from_value(json!({ "data": data, "shape": shape })).expect("Tensor deserialize")
    }

    const ALL_MODES: [ProjectionMode; 4] = [
        ProjectionMode::RateSum,
        ProjectionMode::TemporalHistogram,
        ProjectionMode::MembraneSnapshot,
        ProjectionMode::SpikingTernary,
    ];

    #[test]
    fn reference_adapter_passes_the_conformance_suite_in_every_mode() {
        for mode in ALL_MODES {
            check_domain_adapter_contract(|| PoolResizeTanhAdapter::new(mode))
                .unwrap_or_else(|v| panic!("mode {mode:?}: {v}"));
        }
    }

    #[test]
    fn reference_encode_matches_the_projector_path_bitwise() {
        let mut adapter = PoolResizeTanhAdapter::default();
        let rank2 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0], &[2, 3]);
        let rank1 = Tensor::from_vec(vec![0.5, -0.5], &[2]);
        for (hidden, width) in [
            (&rank2, 3),
            (&rank2, 2),
            (&rank2, 7),
            (&rank1, 5),
            (&rank1, 1),
        ] {
            let got = encode_stimuli_checked(&mut adapter, hidden, width).unwrap();
            let want = projector::embed_to_stimuli_with_width(hidden, width);
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "width {width}"
            );
        }
    }

    #[test]
    fn reference_decode_matches_project_spike_activity() {
        let mut activity = SpikeActivity::from_fired(&[0, 2], 4).unwrap();
        activity.potentials = vec![0.1, 0.2, 0.3, 0.4];
        for mode in ALL_MODES {
            let mut adapter = PoolResizeTanhAdapter::new(mode);
            assert_eq!(adapter.mode(), mode);
            assert_eq!(
                decode_activity_checked(&mut adapter, &activity, 4, 6).unwrap(),
                projector::project_spike_activity(mode, &activity, 4, 6).unwrap(),
                "mode {mode:?}"
            );
        }
    }

    #[test]
    fn encode_input_violations_map_to_documented_errors() {
        let ok = Tensor::from_vec(vec![0.1; 6], &[2, 3]);
        assert!(matches!(
            validate_encode_input(&ok, 0),
            Err(HybridError::ZeroSnnChannels)
        ));
        assert!(matches!(
            validate_encode_input(&ok, MAX_ADAPTER_WIDTH + 1),
            Err(HybridError::InvalidConfig(_))
        ));
        assert!(validate_encode_input(&ok, MAX_ADAPTER_WIDTH).is_ok());
        assert!(matches!(
            validate_encode_input(&Tensor::from_vec(vec![1.0], &[]), 2),
            Err(HybridError::HiddenStateRank { got: 0 })
        ));
        assert!(matches!(
            validate_encode_input(&Tensor::from_vec(vec![0.0; 8], &[2, 2, 2]), 2),
            Err(HybridError::HiddenStateRank { got: 3 })
        ));
        assert!(matches!(
            validate_encode_input(&malformed(vec![], vec![0, 4]), 2),
            Err(HybridError::HiddenStateSeqLen {
                expected: 1,
                got: 0
            })
        ));
        assert!(matches!(
            validate_encode_input(&malformed(vec![], vec![0]), 2),
            Err(HybridError::HiddenStateDim {
                expected: 1,
                got: 0
            })
        ));
        assert!(matches!(
            validate_encode_input(&malformed(vec![0.0; 5], vec![2, 3]), 2),
            Err(HybridError::HiddenStateDataLen {
                expected: 6,
                got: 5
            })
        ));
        assert!(matches!(
            validate_encode_input(&malformed(vec![], vec![usize::MAX, 2]), 2),
            Err(HybridError::HiddenStateDataLen {
                expected: usize::MAX,
                got: 0
            })
        ));
        let mut nan = ok.clone();
        nan.data_mut()[4] = f32::NAN;
        assert!(matches!(
            validate_encode_input(&nan, 2),
            Err(HybridError::NonFinite {
                stage: ForwardValueStage::HiddenState,
                index: 4
            })
        ));
    }

    #[test]
    fn stimuli_and_embedding_output_violations_map_to_documented_errors() {
        assert!(matches!(
            validate_stimuli(&[0.0; 3], 4),
            Err(HybridError::InputLengthMismatch {
                expected: 4,
                got: 3
            })
        ));
        assert!(matches!(
            validate_stimuli(&[0.0, f32::NAN], 2),
            Err(HybridError::NonFinite {
                stage: ForwardValueStage::Stimuli,
                index: 1
            })
        ));
        match validate_stimuli(&[0.0, -1.5], 2) {
            Err(HybridError::InvalidConfig(msg)) => assert!(msg.contains("stimulus 1"), "{msg}"),
            other => panic!("unexpected {other:?}"),
        }
        assert!(validate_stimuli(&[-1.0, 1.0, 0.0], 3).is_ok());
        assert!(matches!(
            validate_embedding(&[0.0], 2),
            Err(HybridError::InputLengthMismatch {
                expected: 2,
                got: 1
            })
        ));
        assert!(matches!(
            validate_embedding(&[f32::INFINITY, 0.0], 2),
            Err(HybridError::NonFinite {
                stage: ForwardValueStage::Embedding,
                index: 0
            })
        ));
        // Unlike stimuli, an embedding is not bound to [-1, 1] by the trait.
        assert!(validate_embedding(&[5.0, -7.0], 2).is_ok());
    }

    #[test]
    fn decode_input_violations_map_to_documented_errors() {
        let act = SpikeActivity::from_fired(&[0], 2).unwrap();
        for (n, d) in [
            (2, 0),
            (0, 4),
            (2, MAX_ADAPTER_WIDTH + 1),
            (MAX_ADAPTER_WIDTH + 1, 4),
        ] {
            assert!(
                matches!(
                    validate_decode_input(&act, n, d),
                    Err(HybridError::InvalidConfig(_))
                ),
                "num_neurons {n}, embed_dim {d}"
            );
        }
        assert!(matches!(
            validate_decode_input(&act, 3, 4),
            Err(HybridError::InputLengthMismatch {
                expected: 3,
                got: 2
            })
        ));
        assert!(validate_decode_input(&act, 2, 4).is_ok());
    }

    /// Finite inputs can still overflow inside the pool/resize average. The
    /// intermediate pooled/resized buffers are checked before `tanh` — which
    /// would squash `±Inf` to `±1` and mask the failure — so every overflow
    /// shape is rejected at the `embedding` stage, never emitted as stimuli.
    #[test]
    fn extreme_finite_hidden_state_is_rejected_not_emitted_as_nan() {
        let mut adapter = PoolResizeTanhAdapter::default();
        // Mixed-sign overflow: `+Inf` and `-Inf` meet in one resize window
        // and would produce `NaN`.
        let mixed = Tensor::from_vec(vec![f32::MAX, -f32::MAX, f32::MAX, -f32::MAX], &[2, 2]);
        // Same-sign overflow inside `mean_pool`: `tanh(+Inf)` would emit a
        // well-formed `1.0` stimulus if the pooled buffer were unchecked.
        let same_sign = Tensor::from_vec(vec![f32::MAX, f32::MAX], &[2, 1]);
        // Finite pooled values can still overflow inside a resize window.
        let resize_overflow = Tensor::from_vec(vec![f32::MAX, f32::MAX], &[2]);
        for hidden in [&mixed, &same_sign, &resize_overflow] {
            assert!(validate_encode_input(hidden, 1).is_ok());
            for result in [
                adapter.encode_stimuli(hidden, 1),
                encode_stimuli_checked(&mut adapter, hidden, 1),
            ] {
                assert!(
                    matches!(
                        result,
                        Err(HybridError::NonFinite {
                            stage: ForwardValueStage::Embedding,
                            index: 0
                        })
                    ),
                    "{result:?}"
                );
            }
        }
    }

    #[test]
    fn declared_metadata_describes_the_reference_adapter() {
        let rate = PoolResizeTanhAdapter::new(ProjectionMode::RateSum);
        assert_eq!(
            rate.encoding(),
            SpikeEncoding::new(
                SpikeEncodingKind::Other("pool_resize_tanh".into()),
                1,
                Some((-1.0, 1.0))
            )
        );
        assert_eq!(rate.decoding().kind, SpikeDecodingKind::Rate);
        assert_eq!(
            PoolResizeTanhAdapter::new(ProjectionMode::MembraneSnapshot)
                .decoding()
                .kind,
            SpikeDecodingKind::Membrane
        );
    }

    /// A hook emitting `-0.5` despite declaring `(0.0, 1.0)` passes the global
    /// `[-1, 1]` bound but fails the checked entry point's declared-range
    /// validation.
    #[test]
    fn checked_encode_enforces_the_declared_encoding_range() {
        struct NarrowRange;
        impl DomainAdapter for NarrowRange {
            fn encode_stimuli(&mut self, _hidden: &Tensor, snn_width: usize) -> Result<Vec<f32>> {
                Ok(vec![-0.5; snn_width])
            }
            fn decode_activity(
                &mut self,
                _activity: &SpikeActivity,
                _num_neurons: usize,
                embed_dim: usize,
            ) -> Result<Vec<f32>> {
                Ok(vec![0.0; embed_dim])
            }
            fn encoding(&self) -> SpikeEncoding {
                SpikeEncoding::new(SpikeEncodingKind::Rate, 1, Some((0.0, 1.0)))
            }
        }
        let hidden = Tensor::from_vec(vec![0.1; 4], &[4]);
        let mut adapter = NarrowRange;
        match encode_stimuli_checked(&mut adapter, &hidden, 2) {
            Err(HybridError::InvalidConfig(msg)) => {
                assert!(msg.contains("declared encoding range"), "{msg}")
            }
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(adapter.encode_stimuli(&hidden, 2).unwrap(), [-0.5; 2]);
        assert!(validate_stimulus_range(&[0.5, 1.0], (0.0, 1.0)).is_ok());
        assert!(validate_stimulus_range(&[-0.1], (0.0, 1.0)).is_err());
    }

    #[test]
    fn adapter_is_object_safe() {
        let mut boxed: Box<dyn DomainAdapter> = Box::new(PoolResizeTanhAdapter::default());
        let hidden = Tensor::from_vec(vec![0.25; 4], &[4]);
        assert_eq!(
            encode_stimuli_checked(&mut *boxed, &hidden, 4)
                .unwrap()
                .len(),
            4
        );
        boxed.reset();
    }
}
