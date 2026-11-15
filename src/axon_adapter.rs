// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`DomainAdapter`] implementations backed by `axon-encoder` (feature
//! `axon-encoder`).
//!
//! `axon-encoder` owns sensory encoding in the rmems stack — rate, temporal,
//! predictive, population, and neuromodulator-driven encoders all live there,
//! not in this crate. [`AxonEncoderAdapter`] bridges one of those encoders to
//! the [`DomainAdapter`] representation boundary so a hybrid pipeline can run
//! rate- or latency-coded stimuli without any in-tree encoder code.
//!
//! Encode: mean-pool the hidden state to `[dim]`, resize to `snn_width`,
//! `tanh` into `[-1, 1]` (the same bounded signal the reference adapter
//! emits), then advance the wrapped [`Encoder`] by exactly one streaming tick
//! via [`Encoder::encode_step`]. Each emitted [`SpikeEvent`] contributes `+1.0`
//! (excitatory) or `-1.0` (inhibitory) to its channel's stimulus; channels
//! saturate at `±`[`STIMULUS_BOUND`] when an encoder emits several spikes per
//! channel in one step. Decode stays in-tree: [`decode_activity`] delegates to
//! [`PoolResizeTanhAdapter`], because `axon-encoder` has no decoders.
//!
//! # Contract notes
//!
//! - **Streaming is the contract.** One `encode_stimuli` call is one
//!   `encode_step`, so stateful encoders (membrane potentials, phase
//!   accumulators, history windows) carry across calls — this is what makes
//!   the adapter stateful rather than a per-call projection.
//! - **Failure atomicity.** Every validation that can reject the call runs
//!   *before* `encode_step`, because encoder state is not snapshotable: a
//!   rejected call leaves the encoder exactly as it was. The single
//!   post-encode error — a spike channel outside `0..snn_width` — cannot be
//!   produced by any conforming `axon-encoder` encoder (channels are input
//!   indices, and the input is exactly `snn_width` wide); it exists to keep a
//!   misbehaving out-of-crate [`Encoder`] from corrupting the stimulus width
//!   silently, and it is the one path on which adapter state may have moved.
//! - **Panics avoided.** `axon-encoder` 0.6 panics on two input shapes that
//!   this adapter rejects first with [`HybridError::InvalidConfig`]:
//!   `EmbeddingRateEncoder` and `DerivativeEncoder` `expect` a `u16` channel
//!   index, so `snn_width` is capped at
//!   [`MAX_ENCODER_CHANNEL_INPUTS`]; and [`PopulationEncoder`] fans each input
//!   out to `num_neurons` spike channels, which can never fit the
//!   `snn_width`-wide stimulus, so wrapping one is rejected at construction
//!   (the [`Encoder`] trait does not report widths, so the check downcasts —
//!   the same approach cortex-tensor's `AxonEncoder` takes).
//! - **Modulated paths.** [`ModulatedEncoder`] gain calls are not part of the
//!   [`DomainAdapter`] signature (the hooks carry no neuromodulator state), so
//!   this adapter never calls them. Callers that need gain modulation drive
//!   [`encoder_mut`](AxonEncoderAdapter::encoder_mut) between host calls and
//!   own the axon-encoder gain names — when the pipeline's modulators are
//!   `neuromod`-side, map names through the explicit RM-1886 mapping rather
//!   than assuming the crates' vocabularies agree.
//!
//! [`decode_activity`]: DomainAdapter::decode_activity
//! [`ModulatedEncoder`]: axon_encoder::ModulatedEncoder

use std::any::Any;

use axon_encoder::Encoder;
use axon_encoder::prelude::{
    EmbeddingEncoderConfig, EmbeddingRateEncoder, PopulationEncoder,
    RateEncoder as AxonRateEncoder, TemporalEncoder,
};

use crate::adapter::{
    DomainAdapter, PoolResizeTanhAdapter, validate_encode_input, validate_stimuli,
};
use crate::projector;
use crate::tensor::Tensor;
use crate::traits::SpikeActivity;
use crate::types::ProjectionMode;
use crate::validate;
use crate::{
    ForwardValueStage, HybridError, Result, STIMULUS_BOUND, SpikeDecoding, SpikeEncoding,
    SpikeEncodingKind,
};

/// Highest input length axon-encoder 0.6 can tag with `u16` spike-channel IDs.
/// `EmbeddingRateEncoder` and `DerivativeEncoder` `expect` the `u16` conversion
/// inside `encode_step`, so a longer input would panic rather than error.
const MAX_ENCODER_CHANNEL_INPUTS: usize = u16::MAX as usize + 1;

/// [`DomainAdapter`] whose ANN → SNN encode runs one `axon-encoder`
/// [`Encoder::encode_step`] tick per call.
///
/// Generic over the encoder so callers choose the scheme; the [`rate`],
/// [`embedding_rate`], and [`temporal`] constructors cover the encoders the
/// stack uses most, and [`with_encoder`] accepts any configured `Encoder`
/// with a caller-declared [`SpikeEncodingKind`].
///
/// [`rate`]: AxonEncoderAdapter::rate
/// [`embedding_rate`]: AxonEncoderAdapter::embedding_rate
/// [`temporal`]: AxonEncoderAdapter::temporal
/// [`with_encoder`]: AxonEncoderAdapter::with_encoder
pub struct AxonEncoderAdapter<E> {
    encoder: E,
    kind: SpikeEncodingKind,
    decode: PoolResizeTanhAdapter,
}

impl<E: Encoder + 'static> AxonEncoderAdapter<E> {
    /// Wrap a configured `axon-encoder` [`Encoder`], declaring `kind` as the
    /// plan-visible [`SpikeEncodingKind`]. `mode` selects the in-tree
    /// decode path (see [`PoolResizeTanhAdapter`]).
    ///
    /// Rejects [`PopulationEncoder`]: it fans each input out to `num_neurons`
    /// spike channels (`input_index * num_neurons + neuron`), so its channel
    /// space can never fit inside the `snn_width`-wide stimulus a
    /// [`DomainAdapter`] must emit.
    pub fn with_encoder(encoder: E, kind: SpikeEncodingKind, mode: ProjectionMode) -> Result<Self> {
        if (&encoder as &dyn Any)
            .downcast_ref::<PopulationEncoder>()
            .is_some()
        {
            return Err(HybridError::InvalidConfig(
                "axon-encoder PopulationEncoder maps each input to num_neurons spike \
                 channels, which cannot fit the snn_width-wide DomainAdapter stimulus; \
                 wrap a single-channel encoder (RateEncoder, LatencyEncoder, \
                 TemporalEncoder, EmbeddingRateEncoder, …)"
                    .into(),
            ));
        }
        Ok(Self {
            encoder,
            kind,
            decode: PoolResizeTanhAdapter::new(mode),
        })
    }

    /// The wrapped encoder (e.g. to inspect its [`Encoder::time_model`]).
    pub fn encoder(&self) -> &E {
        &self.encoder
    }

    /// The wrapped encoder, for callers that drive state directly —
    /// [`ModulatedEncoder::encode_step_with_gains`] for a modulated step, or
    /// an encoder-specific accessor.
    ///
    /// [`ModulatedEncoder::encode_step_with_gains`]: axon_encoder::ModulatedEncoder::encode_step_with_gains
    pub fn encoder_mut(&mut self) -> &mut E {
        &mut self.encoder
    }

    /// Reverse-path projection mode used by
    /// [`decode_activity`](DomainAdapter::decode_activity).
    pub const fn decode_mode(&self) -> ProjectionMode {
        self.decode.mode()
    }
}

impl AxonEncoderAdapter<AxonRateEncoder> {
    /// Default maximum firing rate for [`rate`]: `10 Hz` against the
    /// compatibility `dt_seconds = 0.1` gives at most one spike per tick at
    /// full signal, matching the one-tick-per-forward pacing of the host.
    pub const DEFAULT_MAX_RATE_HZ: f32 = 10.0;

    /// Stateful [`RateEncoder`][AxonRateEncoder] adapter: `base_rate = 0`,
    /// `max_rate` [`DEFAULT_MAX_RATE_HZ`], input range `[-1, 1]` (the `tanh`
    /// signal domain), `dt_seconds` the compatibility `0.1`.
    ///
    /// [`Self::with_encoder`] remains the knob for callers that need other
    /// rates, a caller-owned RNG (`encode_with_rng`), or a different `dt`.
    ///
    /// [`rate`]: AxonEncoderAdapter::rate
    pub fn rate(mode: ProjectionMode) -> Result<Self> {
        let encoder = AxonRateEncoder::try_new(
            0.0,
            Self::DEFAULT_MAX_RATE_HZ,
            (-STIMULUS_BOUND, STIMULUS_BOUND),
            AxonRateEncoder::DEFAULT_DT_SECONDS,
        )
        .map_err(|e| HybridError::InvalidConfig(e.to_string()))?;
        Self::with_encoder(encoder, SpikeEncodingKind::Rate, mode)
    }
}

impl AxonEncoderAdapter<EmbeddingRateEncoder> {
    /// Integrate-and-fire rate adapter over [`EmbeddingRateEncoder`]: each
    /// channel accumulates its input into a persistent membrane potential and
    /// emits a step-start spike when it crosses `v_th`. `num_channels` bounds
    /// which of the `snn_width` signal lanes can ever fire; a `snn_width`
    /// larger than `num_channels` leaves the tail channels silent (the
    /// encoder ignores inputs past its channel count — this adapter does not
    /// error on the mismatch, since silence is representable, but callers
    /// normally size `num_channels == snn_width`).
    pub fn embedding_rate(num_channels: usize, v_th: f32, mode: ProjectionMode) -> Result<Self> {
        let encoder = EmbeddingRateEncoder::try_new(num_channels, EmbeddingEncoderConfig { v_th })
            .map_err(|e| HybridError::InvalidConfig(e.to_string()))?;
        Self::with_encoder(encoder, SpikeEncodingKind::Rate, mode)
    }
}

impl AxonEncoderAdapter<TemporalEncoder> {
    /// Change-threshold temporal adapter over [`TemporalEncoder`]: a channel
    /// fires when its recent signal history crosses a `change_threshold`, at
    /// the latency that threshold carries. `history_depth` must be `>= 6`;
    /// `change_thresholds` are `(threshold, tick)` pairs; `num_channels`
    /// bounds which signal lanes carry history (same sizing rule as
    /// [`embedding_rate`](Self::embedding_rate)).
    ///
    /// Declared kind is `Other("axon_temporal")`: the scheme is
    /// change/threshold-driven, not the signal-proportional latency coding
    /// [`SpikeEncodingKind::Latency`] names.
    pub fn temporal(
        history_depth: usize,
        change_thresholds: Vec<(f32, u16)>,
        num_channels: usize,
        mode: ProjectionMode,
    ) -> Result<Self> {
        let encoder = TemporalEncoder::try_new(history_depth, change_thresholds, num_channels)
            .map_err(|e| HybridError::InvalidConfig(e.to_string()))?;
        Self::with_encoder(
            encoder,
            SpikeEncodingKind::Other("axon_temporal".into()),
            mode,
        )
    }
}

impl<E: Encoder + Send + 'static> DomainAdapter for AxonEncoderAdapter<E> {
    fn encode_stimuli(&mut self, hidden: &Tensor, snn_width: usize) -> Result<Vec<f32>> {
        validate_encode_input(hidden, snn_width)?;
        if snn_width > MAX_ENCODER_CHANNEL_INPUTS {
            return Err(HybridError::InvalidConfig(format!(
                "snn_width {snn_width} exceeds axon-encoder's {MAX_ENCODER_CHANNEL_INPUTS}-channel u16 spike-id space"
            )));
        }
        let pooled = projector::mean_pool(hidden);
        validate::validate_finite(&pooled, ForwardValueStage::Embedding)?;
        let mut signal = projector::resize_to(&pooled, snn_width);
        validate::validate_finite(&signal, ForwardValueStage::Embedding)?;
        for v in &mut signal {
            *v = v.tanh();
        }
        // Commit point: every check above ran before encoder state advances.
        let output = self.encoder.encode_step(&signal);
        let mut stimuli = vec![0.0f32; snn_width];
        for spike in &output.spikes {
            let index = usize::from(spike.channel);
            if index >= snn_width {
                return Err(HybridError::InvalidConfig(format!(
                    "axon-encoder emitted spike channel {index} outside the {snn_width}-wide stimulus"
                )));
            }
            stimuli[index] = (stimuli[index] + if spike.polarity { 1.0 } else { -1.0 })
                .clamp(-STIMULUS_BOUND, STIMULUS_BOUND);
        }
        validate_stimuli(&stimuli, snn_width)?;
        Ok(stimuli)
    }

    fn decode_activity(
        &mut self,
        activity: &SpikeActivity,
        num_neurons: usize,
        embed_dim: usize,
    ) -> Result<Vec<f32>> {
        self.decode
            .decode_activity(activity, num_neurons, embed_dim)
    }

    fn reset(&mut self) {
        self.encoder.reset();
    }

    fn encoding(&self) -> SpikeEncoding {
        SpikeEncoding::new(
            self.kind.clone(),
            1,
            Some((-STIMULUS_BOUND, STIMULUS_BOUND)),
        )
    }

    fn decoding(&self) -> SpikeDecoding {
        self.decode.decoding()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::conformance::check_domain_adapter_contract;

    fn hidden() -> Tensor {
        Tensor::from_vec(vec![0.2, -0.4, 0.7, -0.1, 0.9, 0.0], &[2, 3])
    }

    #[test]
    fn rate_adapter_passes_the_conformance_suite() {
        check_domain_adapter_contract(|| {
            AxonEncoderAdapter::rate(ProjectionMode::RateSum).expect("rate adapter")
        })
        .unwrap_or_else(|v| panic!("{v}"));
    }

    #[test]
    fn embedding_rate_adapter_passes_the_conformance_suite() {
        check_domain_adapter_contract(|| {
            AxonEncoderAdapter::embedding_rate(4, 0.4, ProjectionMode::RateSum)
                .expect("embedding rate adapter")
        })
        .unwrap_or_else(|v| panic!("{v}"));
    }

    #[test]
    fn temporal_adapter_passes_the_conformance_suite() {
        check_domain_adapter_contract(|| {
            AxonEncoderAdapter::temporal(6, vec![(0.1, 0), (0.5, 3)], 4, ProjectionMode::RateSum)
                .expect("temporal adapter")
        })
        .unwrap_or_else(|v| panic!("{v}"));
    }

    #[test]
    fn encode_emits_bounded_stimuli_at_snn_width() {
        let mut adapter =
            AxonEncoderAdapter::embedding_rate(3, 0.05, ProjectionMode::RateSum).unwrap();
        let stimuli = adapter.encode_stimuli(&hidden(), 3).unwrap();
        assert_eq!(stimuli.len(), 3);
        assert!(stimuli.iter().all(|v| v.abs() <= STIMULUS_BOUND));
        // Strong signal: at least one channel must have fired.
        assert!(stimuli.iter().any(|v| *v != 0.0));
    }

    #[test]
    fn encode_step_is_streaming_stateful() {
        let mut adapter =
            AxonEncoderAdapter::embedding_rate(3, 0.9, ProjectionMode::RateSum).unwrap();
        // A sub-threshold first call accumulates but does not fire; repeating
        // the same input must eventually fire — the encoder remembered.
        let first = adapter.encode_stimuli(&hidden(), 3).unwrap();
        let mut fired = first.iter().any(|v| *v != 0.0);
        for _ in 0..16 {
            if adapter
                .encode_stimuli(&hidden(), 3)
                .unwrap()
                .iter()
                .any(|v| *v != 0.0)
            {
                fired = true;
                break;
            }
        }
        assert!(fired, "membrane state should accumulate across calls");
    }

    #[test]
    fn reset_restores_fresh_state() {
        let mut adapter =
            AxonEncoderAdapter::embedding_rate(3, 0.9, ProjectionMode::RateSum).unwrap();
        let fresh = adapter.encode_stimuli(&hidden(), 3).unwrap();
        for _ in 0..8 {
            adapter.encode_stimuli(&hidden(), 3).unwrap();
        }
        adapter.reset();
        let after_reset = adapter.encode_stimuli(&hidden(), 3).unwrap();
        assert_eq!(fresh, after_reset);
    }

    #[test]
    fn population_encoder_is_rejected_at_construction() {
        let encoder = PopulationEncoder::try_new(2, (0.0, 1.0), 0.5).expect("population encoder");
        let result = AxonEncoderAdapter::with_encoder(
            encoder,
            SpikeEncodingKind::Population,
            ProjectionMode::RateSum,
        );
        match result {
            Err(HybridError::InvalidConfig(_)) => {}
            Err(other) => panic!("expected InvalidConfig, got {other}"),
            Ok(_) => panic!("PopulationEncoder must be rejected at construction"),
        }
    }

    #[test]
    fn oversized_snn_width_is_a_typed_error_not_an_encoder_panic() {
        let mut adapter =
            AxonEncoderAdapter::embedding_rate(3, 0.9, ProjectionMode::RateSum).unwrap();
        let err = adapter
            .encode_stimuli(&hidden(), MAX_ENCODER_CHANNEL_INPUTS + 1)
            .expect_err("width beyond u16 channel space must error");
        assert!(matches!(err, HybridError::InvalidConfig(_)));
    }

    #[test]
    fn declared_encoding_and_decoding_flow_to_plan_metadata() {
        let adapter =
            AxonEncoderAdapter::temporal(6, vec![(0.1, 0)], 4, ProjectionMode::MembraneSnapshot)
                .unwrap();
        let encoding = adapter.encoding();
        assert_eq!(encoding.timesteps, 1);
        assert_eq!(encoding.range, Some((-STIMULUS_BOUND, STIMULUS_BOUND)));
        assert_eq!(
            encoding.kind,
            SpikeEncodingKind::Other("axon_temporal".into())
        );
        assert_eq!(adapter.decoding(), adapter.decode.decoding());
    }
}
