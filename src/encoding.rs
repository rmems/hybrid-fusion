// SPDX-License-Identifier: MIT OR Apache-2.0

//! ANN↔SNN boundary encoders and decoders for hybrid-fusion's spiking step
//! contract.
//!
//! Ported from cortex-tensor `src/snn/mod.rs` (issues #61 / RM-2205).
//! cortex-tensor keeps its own twin `SnnEncoder` / `SnnDecoder` traits for
//! its ANN+SNN reference path (decision RM-2206 / RM-2360); the mapping
//! between the two trait sets is documented in
//! `docs/adr/0002-snn-step-contract-parity.md` and enforced by cross-crate
//! parity tests in `smoke/cortex-neuromod/tests/snn_contract_parity.rs`.
//! [`crate::SpikingNetwork`] consumes the stimuli a
//! [`SpikeEncoder`] produces and emits the fired IDs a [`SpikeDecoder`]
//! turns back into ANN-side readouts; [`crate::SpikingExpertRouter`] composes
//! all three.
//!
//! One deliberate change from cortex: [`SpikeDecoder::decode`] is sized by
//! `num_neurons` (the output population), not by the input channel count.
//! hybrid-fusion keeps input width and output population separate
//! ([`crate::SpikingNetwork::num_neurons`]); backends with
//! `num_neurons != num_channels` keep working, and the one-expert-one-channel
//! case is `num_neurons == num_channels == num_experts`.

use serde::{Deserialize, Serialize};

use crate::error::{HybridError, Result};

/// How an ANN-side signal is turned into SNN stimuli.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpikeEncodingKind {
    /// Firing rate proportional to the signal (this crate's `RateEncoder`).
    Rate,
    /// Spike latency inversely proportional to the signal.
    Latency,
    /// Spikes encode the change between consecutive signals.
    Delta,
    /// Signal distributed over a bank of binary channels.
    Population,
    /// Predictive / error-coding encoders.
    Predictive,
    /// Backend-specific scheme named by the encoder.
    Other(String),
}

/// Declared encoder contract, copied onto plan ports (see
/// [`crate::PortSpec::encoding`]).
///
/// `PartialEq`/`Eq` compare the `range` bounds bitwise so the type stays a
/// total equivalence (NaN compares equal to itself with identical bits).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeEncoding {
    /// Encoding scheme.
    pub kind: SpikeEncodingKind,
    /// Stimulus ticks consumed per `encode` call. Every encoder shipped here
    /// reports `1`; the value is the contract hook for multi-tick encodings
    /// (who loops — encoder, host, or backend — is undecided, issue #61 §8).
    pub timesteps: usize,
    /// Stimulus value range, when the encoder bounds one.
    pub range: Option<(f32, f32)>,
}

impl SpikeEncoding {
    /// Build a declared encoding.
    pub const fn new(kind: SpikeEncodingKind, timesteps: usize, range: Option<(f32, f32)>) -> Self {
        Self {
            kind,
            timesteps,
            range,
        }
    }
}

impl Default for SpikeEncoding {
    /// Rate encoding, one timestep, unbounded values (the `RateEncoder`
    /// contract). Matches the [`SpikeEncoder::encoding`] default.
    fn default() -> Self {
        Self {
            kind: SpikeEncodingKind::Rate,
            timesteps: 1,
            range: None,
        }
    }
}

/// Bitwise `PartialEq` so `Eq` is a true equivalence for the float range.
impl PartialEq for SpikeEncoding {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.timesteps == other.timesteps
            && match (self.range, other.range) {
                (None, None) => true,
                (Some((lo, hi)), Some((lo2, hi2))) => {
                    lo.to_bits() == lo2.to_bits() && hi.to_bits() == hi2.to_bits()
                }
                _ => false,
            }
    }
}

impl Eq for SpikeEncoding {}

/// How one tick of fired neuron IDs is turned back into ANN-side readouts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpikeDecodingKind {
    /// Per-neuron spike count (this crate's `SpikeCountDecoder`).
    Count,
    /// Spike count divided by the decode window.
    Rate,
    /// Time-to-first-spike latency.
    FirstSpikeLatency,
    /// Membrane potential at the end of the tick.
    Membrane,
    /// Backend-specific scheme named by the decoder.
    Other(String),
}

/// Declared decoder contract, copied onto plan ports (see
/// [`crate::PortSpec::decoding`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpikeDecoding {
    /// Decoding scheme.
    pub kind: SpikeDecodingKind,
    /// Ticks aggregated per readout. Every decoder shipped here reports `1`.
    pub window: usize,
}

impl SpikeDecoding {
    /// Build a declared decoding.
    pub const fn new(kind: SpikeDecodingKind, window: usize) -> Self {
        Self { kind, window }
    }
}

impl Default for SpikeDecoding {
    /// Spike counting over one tick. Matches the [`SpikeDecoder::decoding`]
    /// default.
    fn default() -> Self {
        Self {
            kind: SpikeDecodingKind::Count,
            window: 1,
        }
    }
}

/// ANN→SNN boundary: map an ANN-side f32 signal into backend stimuli.
///
/// Ported from cortex-tensor `SnnEncoder`. `&mut self` because real encoders
/// are stateful (phase accumulators, streaming windows); one `encode` call is
/// one backend tick. Encoder state advances when `encode` is called — it is
/// NOT failure-atomic with respect to later forward errors; callers needing a
/// clean epoch reset the encoder ([`crate::SpikingExpertRouter::reset`] does
/// so only after the backend reset succeeds).
pub trait SpikeEncoder {
    /// Produce a stimulus vector of exactly `channels` length, or an error
    /// when the encoding cannot be represented in `channels`.
    fn encode(&mut self, ann_signal: &[f32], channels: usize) -> Result<Vec<f32>>;

    /// Restore the encoder's streaming state (phase/window) to a clean epoch.
    /// Stateless encoders use the no-op default.
    fn reset(&mut self) {}

    /// Declared encoding, copied onto the plan port
    /// ([`crate::PortSpec::encoding`]). Default: rate, 1 timestep, unbounded.
    fn encoding(&self) -> SpikeEncoding {
        SpikeEncoding::default()
    }
}

/// SNN→ANN boundary: map one tick of fired neuron IDs into per-neuron f32
/// readouts.
///
/// Ported from cortex-tensor `SnnDecoder`, now **fallible** and sized by
/// `num_neurons` instead of the channel count: readout index `i` is neuron
/// `i` of the output population, and out-of-range fired IDs are an error
/// rather than silently dropped.
pub trait SpikeDecoder {
    /// Produce a readout vector of exactly `num_neurons` length, or an error.
    fn decode(&self, fired: &[usize], num_neurons: usize) -> Result<Vec<f32>>;

    /// Declared decoding, copied onto the plan port
    /// ([`crate::PortSpec::decoding`]). Default: count, 1-tick window.
    fn decoding(&self) -> SpikeDecoding {
        SpikeDecoding::default()
    }
}

/// Linear encoder: scales each input by `gain` and tiles it across the
/// backend's channel count. Expert *i* occupies channel `i % len`, so with
/// `channels >= len` each signal element owns a contiguous prefix of
/// channels. Semantics unchanged from cortex-tensor's `RateEncoder`; this
/// encoder never fails.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateEncoder {
    /// Multiplier applied to every input element.
    pub gain: f32,
}

impl Default for RateEncoder {
    fn default() -> Self {
        Self { gain: 1.0 }
    }
}

impl SpikeEncoder for RateEncoder {
    fn encode(&mut self, ann_signal: &[f32], channels: usize) -> Result<Vec<f32>> {
        let mut stimulus = vec![0.0f32; channels];
        if ann_signal.is_empty() {
            return Ok(stimulus);
        }
        for (idx, slot) in stimulus.iter_mut().enumerate() {
            *slot = ann_signal[idx % ann_signal.len()] * self.gain;
        }
        Ok(stimulus)
    }
}

/// Per-neuron spike-count readout: each neuron's output is the number of
/// times it fired in the tick (0 or 1 for at-most-once fired-ID lists).
///
/// Unlike cortex-tensor's decoder, an out-of-range fired ID is an error
/// ([`HybridError::SnnSpikeOutOfRange`]) instead of being silently dropped:
/// fail closed rather than hide a backend contract violation. Hosts validate
/// the fired list before decoding, so a compliant flow never hits it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpikeCountDecoder;

impl SpikeDecoder for SpikeCountDecoder {
    fn decode(&self, fired: &[usize], num_neurons: usize) -> Result<Vec<f32>> {
        let mut counts = vec![0.0f32; num_neurons];
        for &idx in fired {
            if idx >= num_neurons {
                return Err(HybridError::SnnSpikeOutOfRange {
                    neuron: idx,
                    num_neurons,
                });
            }
            counts[idx] += 1.0;
        }
        Ok(counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_encoder_tiles_signal_across_channels() {
        let mut enc = RateEncoder { gain: 2.0 };
        assert_eq!(
            enc.encode(&[0.5, -1.0], 5).unwrap(),
            vec![1.0, -2.0, 1.0, -2.0, 1.0]
        );
        assert_eq!(enc.encode(&[], 3).unwrap(), vec![0.0; 3]);
        // Gate logits are not pre-bounded: the router path feeds raw logits,
        // so ±Inf and large values pass through for the backend to reject.
        let out = enc.encode(&[f32::INFINITY, 1e30], 2).unwrap();
        assert_eq!(out, vec![f32::INFINITY, 2e30]);
    }

    #[test]
    fn spike_count_decoder_counts_per_neuron() {
        let out = SpikeCountDecoder.decode(&[0, 2], 4).unwrap();
        assert_eq!(out, vec![1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn spike_count_decoder_rejects_out_of_range_fired_ids() {
        // cortex silently ignored index 9 here; fail closed instead.
        assert!(matches!(
            SpikeCountDecoder.decode(&[0, 9], 4),
            Err(HybridError::SnnSpikeOutOfRange {
                neuron: 9,
                num_neurons: 4
            })
        ));
        assert!(SpikeCountDecoder.decode(&[], 4).is_ok());
    }

    #[test]
    fn encoder_and_decoder_declared_contracts_default() {
        assert_eq!(
            RateEncoder::default().encoding(),
            SpikeEncoding::new(SpikeEncodingKind::Rate, 1, None)
        );
        assert_eq!(
            SpikeCountDecoder.decoding(),
            SpikeDecoding::new(SpikeDecodingKind::Count, 1)
        );
    }

    #[test]
    fn spike_encoding_equality_is_bitwise_and_total() {
        let a = SpikeEncoding::new(SpikeEncodingKind::Rate, 1, Some((0.0, 1.0)));
        let b = SpikeEncoding::new(SpikeEncodingKind::Rate, 1, Some((0.0, 1.0)));
        let nan = SpikeEncoding::new(SpikeEncodingKind::Rate, 1, Some((f32::NAN, 1.0)));
        assert_eq!(a, b);
        // NaN equals itself bitwise, so Eq stays reflexive.
        assert_eq!(nan, nan.clone());
        assert_ne!(a, nan);
        assert_ne!(a, SpikeEncoding::new(SpikeEncodingKind::Rate, 2, None));
    }

    #[test]
    fn spike_encoding_serde_round_trip() {
        let enc = SpikeEncoding::new(SpikeEncodingKind::Population, 1, Some((0.0, 1.0)));
        let json = serde_json::to_string(&enc).unwrap();
        assert!(json.contains("\"population\""), "{json}");
        let back: SpikeEncoding = serde_json::from_str(&json).unwrap();
        assert_eq!(back, enc);

        let dec = SpikeDecoding::new(SpikeDecodingKind::FirstSpikeLatency, 1);
        let json = serde_json::to_string(&dec).unwrap();
        assert!(json.contains("\"first_spike_latency\""), "{json}");
        let back: SpikeDecoding = serde_json::from_str(&json).unwrap();
        assert_eq!(back, dec);
    }
}
