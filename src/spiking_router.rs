// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hybrid spiking MoE routing host: gate logits → encode → SNN step → decode
//! → boosted top-k.
//!
//! Ported from cortex-tensor `SpikingMoeRouter` (`src/snn/mod.rs`) as the
//! single owner of spiking MoE routing (issues #61 / RM-2205); the #93 fixes
//! (additive spike boost, bounded hidden gain, validated backend spikes) move
//! with the code so they are ported exactly once.
//!
//! Like [`crate::ReverseHybridPath`], this is a *host*: it composes the
//! [`crate::ExpertRouter`], [`crate::SpikingNetwork`],
//! [`crate::SpikeEncoder`], and [`crate::SpikeDecoder`] contracts and owns no
//! neuron dynamics or checkpoint math. It is deliberately not folded into
//! [`crate::HybridNetwork`]'s generics (AGENTS.md: do not fold hosts into
//! `HybridNetwork`).
//!
//! # Routing contract
//!
//! * **Scores:** `gate_logit + spike_boost * activity`, where `activity` is
//!   the decoder readout for the expert's neuron (≥ 0) and `spike_boost` is
//!   [`DEFAULT_SPIKE_BOOST`] unless changed with
//!   [`SpikingExpertRouter::with_spike_boost`]. The boost is additive so it
//!   is monotone in spike activity for *any* logit sign: a spike can only
//!   promote its expert, never demote it (the earlier multiplicative
//!   `gate * (1 + spikes)` pushed negative logits further down the ranking).
//!   Selection ranks the boosted scores directly (higher score first, lower
//!   expert ID on ties); `expert_weights` is their softmax. Boosted scores
//!   saturate at `f32::MAX` instead of overflowing to `+Inf`; a `±Inf` gate
//!   logit keeps its value (issue #93).
//! * **Hidden gain:** [`SpikingRouteOutput::hidden_gain`] is the selected
//!   softmax mass weighted by activity normalised to the tick's peak, so it
//!   stays within `[0, 1]` whatever the decoder's count scale. When none of
//!   the selected experts is active (a silent tick, or spikes only on
//!   unselected neurons) the gain falls back to the dense selected mass
//!   instead of collapsing to zero. The host reports the gain; the caller
//!   scales its embedding with it (issue #93).
//!
//! # Validation
//!
//! Everything checkable before the SNN tick is checked before it — modulator
//! finiteness, the frozen-evaluation capability, gate-logit count and NaN,
//! and encoder-stimulus NaN — so a rejected forward never consumes a tick and
//! a stateful encoder never advances for a tick the backend then rejects.
//! After the tick, the fired-ID list is checked against the output population
//! (out-of-range → [`HybridError::SnnSpikeOutOfRange`], duplicates →
//! [`HybridError::SnnDuplicateSpike`]) and the readout must have exactly
//! `num_neurons` entries, be NaN-free, and be finite and non-negative
//! ([`HybridError::NonFinite`], [`HybridError::SnnInvalidReadout`]). Errors
//! raised after the tick have consumed it; `reset` is the recovery path on
//! backends that support it.
//!
//! **Neuron → expert mapping.** Readout index `i` is expert `i` for
//! `i < num_experts`. Neurons `num_experts..num_neurons` are excluded from the
//! spike boost and from `hidden_gain` and do not affect routing; they still
//! appear in `fired_neurons` and are still validated. Expert isolation is a
//! wiring property of the backend, not something the host can check: it holds
//! for an identity-wired backend (`NeuromodSnn::identity_wired` under the
//! `neuromod` feature),
//! not for dense hashed weights.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::encoding::{SpikeDecoder, SpikeEncoder};
use crate::error::{ForwardValueStage, HybridError, Result};
use crate::routing::{routing_entropy, softmax, top_k_indices};
use crate::telemetry;
use crate::traits::{ExpertRouteOutput, ExpertRouter, NeuroModulators, SpikingNetwork};

/// Default additive routing boost per unit of decoded spike activity: one
/// spike on a channel adds `1.0` to its expert's gate logit.
pub const DEFAULT_SPIKE_BOOST: f32 = 1.0;

/// One spiking-routing forward: the boosted dense route plus the SNN tick it
/// came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpikingRouteOutput {
    /// Boosted dense route: normalized `expert_weights` (softmax of the
    /// boosted scores), `selected_experts` ranked by those scores, and
    /// optional routing entropy for telemetry.
    pub route: ExpertRouteOutput,
    /// Fired neuron IDs from this tick, each `< num_neurons` and reported at
    /// most once.
    pub fired_neurons: Vec<usize>,
    /// Output population the fired IDs are indices into.
    pub num_neurons: usize,
    /// Selected softmax mass weighted by normalized spike activity, in
    /// `[0, 1]` (issue #93): scale the embedding by this to obtain the
    /// spike-shaped hidden state.
    pub hidden_gain: f32,
}

/// Hybrid MoE routing host through a spiking network:
/// `ExpertRouter` gate logits → [`SpikeEncoder`] → [`SpikingNetwork`] step →
/// [`SpikeDecoder`] → additive spike boost → top-k.
///
/// `R`, `S`, `E`, `D` are generic (not `dyn`) so callers keep concrete
/// backend types for checkpointing and backend-specific tuning.
pub struct SpikingExpertRouter<R, S, E, D>
where
    R: ExpertRouter,
    S: SpikingNetwork,
    E: SpikeEncoder,
    D: SpikeDecoder,
{
    router: R,
    snn: S,
    encoder: E,
    decoder: D,
    spike_boost: f32,
}

/// Reject non-finite modulators before anything advances.
///
/// The SNN trait forwards modulators to the backend, and backends validate
/// them at their own preflight — but that happens *inside* the tick. Checking
/// here keeps a rejected forward from consuming an encoder tick.
fn validate_modulators(modulators: &NeuroModulators) -> Result<()> {
    for (name, value) in [
        ("dopamine", modulators.dopamine),
        ("cortisol", modulators.cortisol),
        ("acetylcholine", modulators.acetylcholine),
        ("tempo", modulators.tempo),
        ("aux_dopamine", modulators.aux_dopamine),
    ] {
        if !value.is_finite() {
            return Err(HybridError::InvalidConfig(format!(
                "SpikingExpertRouter: neuromodulator field '{name}' is non-finite ({value})"
            )));
        }
    }
    Ok(())
}

/// Validate one backend tick against the fired-ID contract: every ID inside
/// `0..num_neurons`, each at most once.
fn validate_fired(fired: &[usize], num_neurons: usize) -> Result<()> {
    let mut seen = HashSet::with_capacity(fired.len());
    for &neuron in fired {
        if neuron >= num_neurons {
            return Err(HybridError::SnnSpikeOutOfRange {
                neuron,
                num_neurons,
            });
        }
        if !seen.insert(neuron) {
            return Err(HybridError::SnnDuplicateSpike { neuron });
        }
    }
    Ok(())
}

/// Validate a decoder readout: exactly `num_neurons` entries, each finite and
/// non-negative. NaN keeps the structured [`HybridError::NonFinite`] report;
/// `±Inf` and negatives are activity values no decoder may emit
/// (issue #93).
fn validate_readout(readout: &[f32], num_neurons: usize) -> Result<()> {
    if readout.len() != num_neurons {
        return Err(HybridError::InputLengthMismatch {
            expected: num_neurons,
            got: readout.len(),
        });
    }
    for (index, &value) in readout.iter().enumerate() {
        if value.is_nan() {
            return Err(HybridError::NonFinite {
                stage: ForwardValueStage::DecoderReadout,
                index,
            });
        }
        if !value.is_finite() || value < 0.0 {
            return Err(HybridError::SnnInvalidReadout { index, value });
        }
    }
    Ok(())
}

impl<R, S, E, D> SpikingExpertRouter<R, S, E, D>
where
    R: ExpertRouter,
    S: SpikingNetwork,
    E: SpikeEncoder,
    D: SpikeDecoder,
{
    /// Compose a router, an SNN backend, and the boundary encoder/decoder.
    ///
    /// Requires `snn.num_neurons() >= router.num_experts()`: readout index
    /// `i` is expert `i`, so a smaller population would silently leave
    /// trailing experts permanently ANN-only. The failure is
    /// [`HybridError::ConfigMismatch`] on
    /// [`HybridError::FIELD_SNN_NUM_NEURONS`] (`configured` = expert count,
    /// `backend` = population).
    ///
    /// # Errors
    ///
    /// - [`HybridError::InvalidConfig`] if the router reports zero experts.
    /// - [`HybridError::ConfigMismatch`] if the output population cannot
    ///   cover the expert set.
    pub fn new(router: R, snn: S, encoder: E, decoder: D) -> Result<Self> {
        let num_experts = router.num_experts();
        if num_experts == 0 {
            return Err(HybridError::InvalidConfig(
                "SpikingExpertRouter: ExpertRouter reported zero experts".into(),
            ));
        }
        let num_neurons = snn.num_neurons();
        if num_neurons < num_experts {
            return Err(HybridError::config_mismatch(
                HybridError::FIELD_SNN_NUM_NEURONS,
                num_experts,
                num_neurons,
            ));
        }
        Ok(Self {
            router,
            snn,
            encoder,
            decoder,
            spike_boost: DEFAULT_SPIKE_BOOST,
        })
    }

    /// Set the additive routing boost per unit of decoded spike activity
    /// (default [`DEFAULT_SPIKE_BOOST`]). Must be finite and non-negative so
    /// the boost stays monotone; `0.0` makes routing purely ANN-driven while
    /// the backend still steps and shapes the hidden gain. For a finite gate
    /// logit the score `gate + spike_boost * activity` saturates at `f32::MAX`
    /// instead of overflowing to `+Inf`; a `±Inf` logit keeps its value, so
    /// the boost never turns a finite score infinite or a `-Inf` logit into
    /// NaN (issue #93).
    ///
    /// # Errors
    ///
    /// [`HybridError::InvalidConfig`] for a non-finite or negative boost.
    pub fn with_spike_boost(mut self, spike_boost: f32) -> Result<Self> {
        if !spike_boost.is_finite() || spike_boost < 0.0 {
            return Err(HybridError::InvalidConfig(format!(
                "spike_boost must be finite and >= 0, got {spike_boost}"
            )));
        }
        self.spike_boost = spike_boost;
        Ok(self)
    }

    /// Additive routing boost per unit of decoded spike activity.
    pub fn spike_boost(&self) -> f32 {
        self.spike_boost
    }

    /// Immutable reference to the composed dense router.
    pub fn router(&self) -> &R {
        &self.router
    }

    /// Mutable reference to the composed dense router.
    pub fn router_mut(&mut self) -> &mut R {
        &mut self.router
    }

    /// Immutable reference to the spiking backend.
    pub fn snn(&self) -> &S {
        &self.snn
    }

    /// Mutable reference to the spiking backend (e.g. to `reseed` it).
    pub fn snn_mut(&mut self) -> &mut S {
        &mut self.snn
    }

    /// Immutable reference to the ANN→SNN encoder.
    pub fn encoder(&self) -> &E {
        &self.encoder
    }

    /// Mutable reference to the ANN→SNN encoder (e.g. to reset its streaming
    /// state after an error).
    pub fn encoder_mut(&mut self) -> &mut E {
        &mut self.encoder
    }

    /// Immutable reference to the SNN→ANN decoder.
    pub fn decoder(&self) -> &D {
        &self.decoder
    }

    /// Learning-enabled forward: backend steps retain plasticity.
    ///
    /// # Errors
    ///
    /// See the [module documentation](self) for the full pre-tick/post-tick
    /// error table; every pre-tick failure leaves backend and encoder state
    /// untouched.
    pub fn forward(
        &mut self,
        embedding: &[f32],
        modulators: &NeuroModulators,
    ) -> Result<SpikingRouteOutput> {
        self.forward_impl(embedding, modulators, false)
    }

    /// Held-out forward: the backend runs its frozen evaluation path
    /// ([`SpikingNetwork::step_frozen`]).
    ///
    /// Fails **before** the encoder advances unless the backend's
    /// [`SpikingNetwork::capabilities`] advertises the `frozen_evaluation`
    /// feature; it never falls back to the learning path (issue #92).
    pub fn forward_frozen(
        &mut self,
        embedding: &[f32],
        modulators: &NeuroModulators,
    ) -> Result<SpikingRouteOutput> {
        self.forward_impl(embedding, modulators, true)
    }

    /// Reset the backend epoch, then the encoder epoch; the dense router's
    /// ANN state is untouched. The encoder resets only if the backend reset
    /// succeeded — a backend that cannot reset keeps a coherent encoder.
    ///
    /// # Errors
    ///
    /// Whatever [`SpikingNetwork::reset`] returns (the trait default is an
    /// unsupported-operation error).
    pub fn reset(&mut self) -> Result<()> {
        self.snn.reset()?;
        self.encoder.reset();
        Ok(())
    }

    /// Shared forward path. `frozen` selects
    /// [`SpikingNetwork::step_frozen`] after the capability check.
    fn forward_impl(
        &mut self,
        embedding: &[f32],
        modulators: &NeuroModulators,
        frozen: bool,
    ) -> Result<SpikingRouteOutput> {
        // ── Pre-tick: nothing may advance for a tick that will be rejected ──
        validate_modulators(modulators)?;
        if frozen {
            let caps = self.snn.capabilities();
            let advertised = caps
                .features
                .iter()
                .any(|feature| feature.as_str() == "frozen_evaluation");
            if !advertised {
                return Err(HybridError::InvalidConfig(format!(
                    "spiking backend '{}' does not advertise frozen_evaluation; \
                     held-out routing never falls back to the learning path",
                    caps.backend_name
                )));
            }
        }
        let num_neurons = self.snn.num_neurons();
        let num_experts = self.router.num_experts();
        let gate_logits = self.router.gate_logits(embedding)?;
        if gate_logits.len() != num_experts {
            return Err(HybridError::InputLengthMismatch {
                expected: num_experts,
                got: gate_logits.len(),
            });
        }
        // Fail closed before the backend consumes a tick.
        if let Some(expert) = gate_logits.iter().position(|s| s.is_nan()) {
            return Err(HybridError::InvalidConfig(format!(
                "SpikingExpertRouter: gate logit for expert {expert} is NaN; \
                 NaN scores are rejected, not sanitized"
            )));
        }
        let stimulus = self.encoder.encode(&gate_logits, self.snn.num_channels())?;
        // Encoders may inject NaN stimuli; reject before the backend consumes
        // a tick so the whole forward stays fail-closed. ±Inf is rankable per
        // the finite-value policy and passes through to the backend, which
        // rejects it in its own failure-atomic preflight.
        if let Some(index) = stimulus.iter().position(|v| v.is_nan()) {
            return Err(HybridError::NonFinite {
                stage: ForwardValueStage::EncoderStimulus,
                index,
            });
        }

        // ── The tick. Backend/runtime failures are reported when the
        // `sentry` feature is on. ──────────────────────────────────────────
        let fired_neurons = match if frozen {
            self.snn.step_frozen(&stimulus, modulators)
        } else {
            self.snn.step(&stimulus, modulators)
        } {
            Ok(fired) => fired,
            Err(err) => {
                telemetry::capture_error(&err);
                return Err(err);
            }
        };

        // ── Post-tick: failures past this point have consumed the tick;
        // `reset` is the recovery path on backends that support it. ────────
        validate_fired(&fired_neurons, num_neurons)?;
        let readout = self.decoder.decode(&fired_neurons, num_neurons)?;
        validate_readout(&readout, num_neurons)?;

        // Additive boost: monotone in activity for any logit sign (issue #93).
        // Both factors are finite and >= 0, so the bonus is finite once its
        // product is capped; a finite logit plus that bonus is then saturated
        // at f32::MAX so it can never become +Inf, while a ±Inf logit keeps
        // its sign (bonus >= 0, so -Inf + bonus cannot be NaN).
        let routing_scores: Vec<f32> = (0..num_experts)
            .map(|expert_id| {
                let gate = gate_logits[expert_id];
                let bonus = (self.spike_boost * readout[expert_id]).min(f32::MAX);
                if gate.is_finite() {
                    (gate + bonus).min(f32::MAX)
                } else {
                    gate
                }
            })
            .collect();
        let expert_weights = softmax(&routing_scores)?;
        // Selection ranks the boosted scores (higher first, lower expert ID
        // on ties), not the softmax weights: after max-subtraction two
        // visibly different scores can underflow to the same weight, and
        // ranking those would pick the wrong expert.
        let selected_experts = top_k_indices(&routing_scores, self.router.top_k());
        let entropy = routing_entropy(&expert_weights);

        // Hidden gain = selected softmax mass weighted by activity normalised
        // to the tick's peak, so it stays within [0, 1] regardless of the
        // decoder's count scale. When no selected expert is active (silent
        // tick, or spikes only on unselected neurons) the gain falls back to
        // the dense selected mass rather than collapsing to zero (issue #93).
        let peak_activity = readout[..num_experts]
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        let selected_active = selected_experts.iter().any(|&idx| readout[idx] > 0.0);
        let hidden_gain: f32 = selected_experts
            .iter()
            .map(|&idx| {
                let activity = if selected_active {
                    readout[idx] / peak_activity
                } else {
                    1.0
                };
                expert_weights[idx] * activity
            })
            .sum();

        Ok(SpikingRouteOutput {
            route: ExpertRouteOutput {
                expert_weights,
                selected_experts,
                routing_entropy: Some(entropy),
            },
            fired_neurons,
            num_neurons,
            hidden_gain,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::{RateEncoder, SpikeCountDecoder};
    use crate::error::HybridError;
    use crate::traits::{ExpertRouteOutput, ExpertRouter, SpikingNetwork};

    /// Issue #93 repro: eight experts with strictly negative gate logits.
    const NEGATIVE_LOGITS: [f32; 8] = [-1.0, -1.25, -1.5, -1.75, -2.0, -2.25, -2.5, -2.75];

    /// Dense router that replays scripted gate logits, mirroring
    /// cortex-tensor's `DenseSim` stub: `route` and `gate_logits` agree.
    struct ScriptedRouter {
        num_experts: usize,
        top_k: usize,
        logits: Vec<f32>,
    }

    impl ScriptedRouter {
        fn new(num_experts: usize, top_k: usize, logits: Vec<f32>) -> Self {
            Self {
                num_experts,
                top_k,
                logits,
            }
        }
    }

    impl ExpertRouter for ScriptedRouter {
        fn num_experts(&self) -> usize {
            self.num_experts
        }
        fn top_k(&self) -> usize {
            self.top_k
        }
        fn route(&mut self, embedding: &[f32]) -> Result<ExpertRouteOutput> {
            let logits = self.gate_logits(embedding)?;
            let weights = softmax(&logits)?;
            let entropy = routing_entropy(&weights);
            Ok(ExpertRouteOutput {
                expert_weights: weights,
                selected_experts: top_k_indices(&logits, self.top_k),
                routing_entropy: Some(entropy),
            })
        }
        fn gate_logits(&mut self, _embedding: &[f32]) -> Result<Vec<f32>> {
            Ok(self.logits.clone())
        }
    }

    /// Router that validates the embedding length the way a real
    /// checkpoint-backed router does.
    struct DimCheckedRouter {
        num_experts: usize,
        top_k: usize,
        expected_dim: usize,
    }

    impl ExpertRouter for DimCheckedRouter {
        fn num_experts(&self) -> usize {
            self.num_experts
        }
        fn top_k(&self) -> usize {
            self.top_k
        }
        fn route(&mut self, embedding: &[f32]) -> Result<ExpertRouteOutput> {
            let logits = self.gate_logits(embedding)?;
            let weights = softmax(&logits)?;
            let entropy = routing_entropy(&weights);
            Ok(ExpertRouteOutput {
                expert_weights: weights,
                selected_experts: top_k_indices(&logits, self.top_k),
                routing_entropy: Some(entropy),
            })
        }
        fn gate_logits(&mut self, embedding: &[f32]) -> Result<Vec<f32>> {
            if embedding.len() != self.expected_dim {
                return Err(HybridError::InputLengthMismatch {
                    expected: self.expected_dim,
                    got: embedding.len(),
                });
            }
            Ok(vec![0.0; self.num_experts])
        }
    }

    /// Deterministic in-tree backend for contract tests. Spikes every neuron
    /// whose stimulus exceeds `threshold`; tracks step counts to prove state
    /// advances and reset works. Not neuron dynamics — the test boundary.
    struct MockSnn {
        channels: usize,
        neurons: usize,
        threshold: f32,
        steps: usize,
        frozen_steps: usize,
        frozen_capability: bool,
    }

    impl MockSnn {
        fn new(channels: usize, neurons: usize, threshold: f32) -> Self {
            Self {
                channels,
                neurons,
                threshold,
                steps: 0,
                frozen_steps: 0,
                frozen_capability: true,
            }
        }

        fn without_frozen_capability(mut self) -> Self {
            self.frozen_capability = false;
            self
        }
    }

    impl SpikingNetwork for MockSnn {
        fn step(&mut self, stimuli: &[f32], _modulators: &NeuroModulators) -> Result<Vec<usize>> {
            assert_eq!(stimuli.len(), self.channels);
            self.steps += 1;
            Ok(stimuli
                .iter()
                .enumerate()
                .filter(|(_, v)| **v > self.threshold)
                .map(|(idx, _)| idx)
                .collect())
        }

        fn step_frozen(
            &mut self,
            stimuli: &[f32],
            _modulators: &NeuroModulators,
        ) -> Result<Vec<usize>> {
            assert_eq!(stimuli.len(), self.channels);
            self.frozen_steps += 1;
            Ok(Vec::new())
        }

        fn num_channels(&self) -> usize {
            self.channels
        }

        fn num_neurons(&self) -> usize {
            self.neurons
        }

        fn reset(&mut self) -> Result<()> {
            self.steps = 0;
            self.frozen_steps = 0;
            Ok(())
        }

        fn capabilities(&self) -> crate::BackendCapabilities {
            let mut features = vec![crate::capabilities::RequiredFeature::new("step")];
            if self.frozen_capability {
                features.push(crate::capabilities::RequiredFeature::new(
                    "frozen_evaluation",
                ));
            }
            crate::BackendCapabilities::snn("mock")
                .with_dtypes([crate::Dtype::F32])
                .with_channels(self.channels)
                .with_num_neurons(self.neurons)
                .with_max_sequence(1)
                .with_features(features)
        }
    }

    /// Backend that replays a fixed fired-ID list every tick, so tests can
    /// script exactly which neurons fire (including out-of-range and
    /// duplicate indices a misbehaving backend might emit). Not neuron
    /// dynamics — the test boundary.
    struct ScriptedSnn {
        channels: usize,
        neurons: usize,
        spikes: Vec<usize>,
    }

    impl ScriptedSnn {
        fn new(neurons: usize, spikes: Vec<usize>) -> Self {
            Self {
                channels: neurons,
                neurons,
                spikes,
            }
        }
    }

    impl SpikingNetwork for ScriptedSnn {
        fn step(&mut self, stimuli: &[f32], _modulators: &NeuroModulators) -> Result<Vec<usize>> {
            assert_eq!(stimuli.len(), self.channels);
            Ok(self.spikes.clone())
        }

        fn step_frozen(
            &mut self,
            stimuli: &[f32],
            modulators: &NeuroModulators,
        ) -> Result<Vec<usize>> {
            self.step(stimuli, modulators)
        }

        fn num_channels(&self) -> usize {
            self.channels
        }

        fn num_neurons(&self) -> usize {
            self.neurons
        }

        fn capabilities(&self) -> crate::BackendCapabilities {
            crate::BackendCapabilities::snn("scripted")
                .with_dtypes([crate::Dtype::F32])
                .with_channels(self.channels)
                .with_num_neurons(self.neurons)
                .with_max_sequence(1)
                .with_features([
                    crate::capabilities::RequiredFeature::new("step"),
                    crate::capabilities::RequiredFeature::new("frozen_evaluation"),
                ])
        }
    }

    /// Decoder that multiplies each spike count by a large factor, standing in
    /// for multi-count readouts (burst counts, rate estimates) whose magnitude
    /// is unbounded from the router's point of view.
    struct ScaledCountDecoder(f32);

    impl SpikeDecoder for ScaledCountDecoder {
        fn decode(&self, fired: &[usize], num_neurons: usize) -> Result<Vec<f32>> {
            Ok(SpikeCountDecoder
                .decode(fired, num_neurons)?
                .into_iter()
                .map(|count| count * self.0)
                .collect())
        }
    }

    /// Decoder that emits a fixed readout regardless of spikes, used to feed
    /// the router readouts (negative, non-finite) a well-behaved decoder
    /// never would.
    struct FixedReadoutDecoder(Vec<f32>);

    impl SpikeDecoder for FixedReadoutDecoder {
        fn decode(&self, _fired: &[usize], _num_neurons: usize) -> Result<Vec<f32>> {
            Ok(self.0.clone())
        }
    }

    /// Encoder that counts `encode` calls, to pin the pre-tick ordering:
    /// a rejected forward must not advance a stateful encoder.
    struct CountingEncoder {
        inner: RateEncoder,
        calls: usize,
        resets: usize,
    }

    impl CountingEncoder {
        fn new(gain: f32) -> Self {
            Self {
                inner: RateEncoder { gain },
                calls: 0,
                resets: 0,
            }
        }
    }

    impl SpikeEncoder for CountingEncoder {
        fn encode(&mut self, ann_signal: &[f32], channels: usize) -> Result<Vec<f32>> {
            self.calls += 1;
            self.inner.encode(ann_signal, channels)
        }
        fn reset(&mut self) {
            self.resets += 1;
        }
    }

    /// Backend whose reset always fails, to pin reset error propagation.
    struct ResetFailsSnn {
        channels: usize,
        neurons: usize,
    }

    impl SpikingNetwork for ResetFailsSnn {
        fn step(&mut self, _stimuli: &[f32], _m: &NeuroModulators) -> Result<Vec<usize>> {
            Ok(Vec::new())
        }
        fn num_channels(&self) -> usize {
            self.channels
        }
        fn num_neurons(&self) -> usize {
            self.neurons
        }
        // Trait default reset: unsupported-operation error.
    }

    fn spiking_router<S: SpikingNetwork>(
        snn: S,
    ) -> SpikingExpertRouter<ScriptedRouter, S, RateEncoder, SpikeCountDecoder> {
        SpikingExpertRouter::new(
            ScriptedRouter::new(8, 2, vec![0.0; 8]),
            snn,
            RateEncoder::default(),
            SpikeCountDecoder,
        )
        .unwrap()
    }

    fn scripted_router<D: SpikeDecoder>(
        spikes: Vec<usize>,
        decoder: D,
    ) -> SpikingExpertRouter<ScriptedRouter, ScriptedSnn, RateEncoder, D> {
        SpikingExpertRouter::new(
            ScriptedRouter::new(8, 2, NEGATIVE_LOGITS.to_vec()),
            ScriptedSnn::new(8, spikes),
            RateEncoder::default(),
            decoder,
        )
        .unwrap()
    }

    #[test]
    fn spiking_forward_routes_and_steps_backend() {
        let mut model = spiking_router(MockSnn::new(8, 8, 0.0));
        let out = model
            .forward(&[0.1, 0.2, 0.3], &NeuroModulators::default())
            .unwrap();
        assert_eq!(out.num_neurons, 8);
        assert_eq!(out.route.selected_experts.len(), 2);
        assert!(out.route.expert_weights.iter().all(|w| w.is_finite()));
        assert_eq!(model.snn().steps, 1);
    }

    #[test]
    fn spiking_forward_rejects_nan_gate_logits_before_backend_step() {
        let mut logits = vec![0.0f32; 8];
        logits[3] = f32::NAN;
        let mut model = SpikingExpertRouter::new(
            ScriptedRouter::new(8, 2, logits),
            MockSnn::new(8, 8, 0.0),
            RateEncoder::default(),
            SpikeCountDecoder,
        )
        .unwrap();
        let err = model
            .forward(&[1.0; 4], &NeuroModulators::default())
            .unwrap_err();
        match err {
            HybridError::InvalidConfig(msg) => assert!(msg.contains("expert 3"), "{msg}"),
            other => panic!("unexpected {other:?}"),
        }
        // Fail-closed: no tick consumed.
        assert_eq!(model.snn().steps, 0);
    }

    #[test]
    fn constructor_rejects_smaller_population_than_experts() {
        assert!(matches!(
            SpikingExpertRouter::new(
                ScriptedRouter::new(8, 2, vec![0.0; 8]),
                MockSnn::new(8, 4, 0.0),
                RateEncoder::default(),
                SpikeCountDecoder
            ),
            Err(HybridError::ConfigMismatch {
                field: HybridError::FIELD_SNN_NUM_NEURONS,
                configured: 8,
                backend: 4
            })
        ));
    }

    #[test]
    fn constructor_rejects_zero_experts() {
        struct ZeroRouter;
        impl ExpertRouter for ZeroRouter {
            fn num_experts(&self) -> usize {
                0
            }
            fn top_k(&self) -> usize {
                1
            }
            fn route(&mut self, _embedding: &[f32]) -> Result<ExpertRouteOutput> {
                unreachable!("num_experts is zero")
            }
        }
        assert!(matches!(
            SpikingExpertRouter::new(
                ZeroRouter,
                MockSnn::new(4, 4, 0.0),
                RateEncoder::default(),
                SpikeCountDecoder
            ),
            Err(HybridError::InvalidConfig(_))
        ));
    }

    #[test]
    fn nan_stimulus_rejected_before_backend_step() {
        let mut model = SpikingExpertRouter::new(
            ScriptedRouter::new(8, 2, NEGATIVE_LOGITS.to_vec()),
            MockSnn::new(8, 8, 0.0),
            CountingEncoder::new(f32::NAN),
            SpikeCountDecoder,
        )
        .unwrap();
        assert!(matches!(
            model
                .forward(&[1.0; 4], &NeuroModulators::default())
                .unwrap_err(),
            HybridError::NonFinite {
                stage: ForwardValueStage::EncoderStimulus,
                ..
            }
        ));
        // Fail-closed: no tick consumed.
        assert_eq!(model.snn().steps, 0);
    }

    #[test]
    fn forward_frozen_uses_frozen_path_and_reset_clears_state() {
        let mut model = spiking_router(MockSnn::new(8, 8, 0.0));
        let _ = model
            .forward(&[1.0; 4], &NeuroModulators::default())
            .unwrap();
        let _ = model
            .forward_frozen(&[1.0; 4], &NeuroModulators::default())
            .unwrap();
        assert_eq!(model.snn().steps, 1);
        assert_eq!(model.snn().frozen_steps, 1);
        assert!(
            model
                .snn()
                .capabilities()
                .features
                .iter()
                .any(|f| f.as_str() == "frozen_evaluation")
        );
        model.reset().unwrap();
        assert_eq!(model.snn().steps, 0);
        assert_eq!(model.snn().frozen_steps, 0);
    }

    /// Issue #92: a held-out forward must never silently run the learning
    /// path. Without the `frozen_evaluation` capability the call fails
    /// before the encoder advances.
    #[test]
    fn forward_frozen_fails_closed_without_frozen_capability() {
        let mut model = SpikingExpertRouter::new(
            ScriptedRouter::new(8, 2, NEGATIVE_LOGITS.to_vec()),
            MockSnn::new(8, 8, 0.0).without_frozen_capability(),
            CountingEncoder::new(1.0),
            SpikeCountDecoder,
        )
        .unwrap();
        let err = model
            .forward_frozen(&[1.0; 4], &NeuroModulators::default())
            .unwrap_err();
        match err {
            HybridError::InvalidConfig(msg) => {
                assert!(msg.contains("frozen_evaluation"), "{msg}")
            }
            other => panic!("unexpected {other:?}"),
        }
        // Fail-closed: neither path stepped and the encoder never advanced.
        assert_eq!(model.snn().steps, 0);
        assert_eq!(model.snn().frozen_steps, 0);
        assert_eq!(model.encoder().calls, 0);
    }

    #[test]
    fn spiking_forward_rejects_wrong_embedding_length() {
        let mut model = SpikingExpertRouter::new(
            DimCheckedRouter {
                num_experts: 8,
                top_k: 2,
                expected_dim: 4,
            },
            MockSnn::new(8, 8, 0.0),
            RateEncoder::default(),
            SpikeCountDecoder,
        )
        .unwrap();
        assert!(matches!(
            model
                .forward(&[0.0; 8], &NeuroModulators::default())
                .unwrap_err(),
            HybridError::InputLengthMismatch {
                expected: 4,
                got: 8
            }
        ));
        assert_eq!(model.snn().steps, 0);
    }

    /// Issue #93 (1): a spike must never demote its expert. With the old
    /// `gate * (1 + spikes)` boost, spiking channel 0 on negative logits
    /// evicted expert 0 (`[0, 1]` → `[1, 2]`).
    #[test]
    fn spike_boost_promotes_spiked_expert_with_negative_gate_logits() {
        let mods = NeuroModulators::default();
        // Silent backend: ranking is the dense ranking.
        let mut silent = scripted_router(Vec::new(), SpikeCountDecoder);
        assert_eq!(
            silent
                .forward(&[0.5; 4], &mods)
                .unwrap()
                .route
                .selected_experts,
            vec![0, 1]
        );

        // Spiking the already-top expert keeps it on top.
        let mut spike_top = scripted_router(vec![0], SpikeCountDecoder);
        assert_eq!(
            spike_top
                .forward(&[0.5; 4], &mods)
                .unwrap()
                .route
                .selected_experts,
            vec![0, 1]
        );

        // Spiking a lower-ranked expert promotes it: -1.5 + 1.0 beats -1.0.
        let mut spike_mid = scripted_router(vec![2], SpikeCountDecoder);
        assert_eq!(
            spike_mid
                .forward(&[0.5; 4], &mods)
                .unwrap()
                .route
                .selected_experts,
            vec![2, 0]
        );
    }

    /// Issue #93 (1): the boost is monotone in the spike count for either
    /// logit sign — more spikes never lower an expert's rank.
    #[test]
    fn spike_boost_is_monotone_in_spike_count_regardless_of_logit_sign() {
        let mods = NeuroModulators::default();
        // Same spread shifted into positive territory (2.75 … 1.0) keeps
        // expert 7 weakest.
        for logits in [NEGATIVE_LOGITS, NEGATIVE_LOGITS.map(|v| v + 3.75)] {
            let mut previous_rank = usize::MAX;
            // Expert 7 holds the lowest logit of each set; raise its readout
            // from 0 to 4 "spikes" via a scaled decoder and watch its rank
            // only improve.
            for count in 0..5 {
                let mut model = SpikingExpertRouter::new(
                    ScriptedRouter::new(8, 2, logits.to_vec()),
                    ScriptedSnn::new(8, vec![7]),
                    RateEncoder::default(),
                    ScaledCountDecoder(count as f32),
                )
                .unwrap();
                let out = model.forward(&[0.5; 4], &mods).unwrap();
                let rank = out
                    .route
                    .selected_experts
                    .iter()
                    .position(|&e| e == 7)
                    .unwrap_or(out.route.selected_experts.len());
                assert!(
                    rank <= previous_rank,
                    "count {count}: rank {rank} > {previous_rank}"
                );
                previous_rank = rank;
            }
            // Four spikes lift the weakest expert (-2.75 or 1.0) to the top.
            assert_eq!(previous_rank, 0, "logits {logits:?}");
        }
    }

    /// Issue #93 (1): `with_spike_boost` scales the additive boost and rejects
    /// values that would break monotonicity.
    #[test]
    fn spike_boost_gain_is_configurable_and_validated() {
        let mods = NeuroModulators::default();
        // beta = 0.1: -1.5 + 0.1 does not beat -1.25, so spiking expert 2
        // changes nothing.
        let mut weak = SpikingExpertRouter::new(
            ScriptedRouter::new(8, 2, NEGATIVE_LOGITS.to_vec()),
            ScriptedSnn::new(8, vec![2]),
            RateEncoder::default(),
            SpikeCountDecoder,
        )
        .unwrap()
        .with_spike_boost(0.1)
        .unwrap();
        assert_eq!(weak.spike_boost(), 0.1);
        assert_eq!(
            weak.forward(&[0.5; 4], &mods)
                .unwrap()
                .route
                .selected_experts,
            vec![0, 1]
        );

        for bad in [-1.0, f32::NAN, f32::INFINITY] {
            let model = scripted_router(Vec::new(), SpikeCountDecoder);
            assert!(matches!(
                model.with_spike_boost(bad),
                Err(HybridError::InvalidConfig(_))
            ));
        }
    }

    /// Issue #93 (2): `hidden_gain` is the selected softmax mass weighted by
    /// normalized spike activity, so it stays in `[0, 1]` even when the
    /// decoder reports huge counts, and a silent tick falls back to the dense
    /// selected mass instead of zeroing the hidden state.
    #[test]
    fn hidden_gain_is_bounded_and_silent_tick_keeps_dense_mass() {
        let mods = NeuroModulators::default();
        // Silent backend: same scale as a dense route (sum of selected
        // softmax weights).
        let mut silent = scripted_router(Vec::new(), SpikeCountDecoder);
        let out = silent.forward(&[0.5; 4], &mods).unwrap();
        let dense_mass: f32 = out
            .route
            .selected_experts
            .iter()
            .map(|&e| out.route.expert_weights[e])
            .sum();
        assert!(dense_mass > 0.0 && dense_mass <= 1.0);
        assert_eq!(out.hidden_gain, dense_mass);

        // Huge multi-count readout on both selected experts: gain is still
        // <= 1.
        let mut loud = scripted_router(vec![0, 1], ScaledCountDecoder(1e6));
        let out = loud.forward(&[0.5; 4], &mods).unwrap();
        assert_eq!(out.route.selected_experts, vec![0, 1]);
        let mass: f32 = out
            .route
            .selected_experts
            .iter()
            .map(|&e| out.route.expert_weights[e])
            .sum();
        assert_eq!(out.hidden_gain, mass);
        assert!(out.hidden_gain.is_finite() && out.hidden_gain <= 1.0);

        // One selected expert spiking, the other silent: only the spiking
        // expert's softmax weight contributes.
        let mut partial = scripted_router(vec![0], SpikeCountDecoder);
        let out = partial.forward(&[0.5; 4], &mods).unwrap();
        assert_eq!(out.route.selected_experts, vec![0, 1]);
        assert_eq!(out.hidden_gain, out.route.expert_weights[0]);
    }

    /// Issue #93 (3): backend fired IDs outside `0..num_neurons` are rejected
    /// before decoding instead of being silently dropped by the decoder.
    #[test]
    fn forward_rejects_backend_spike_outside_population() {
        let mut model = scripted_router(vec![0, 9], SpikeCountDecoder);
        assert!(matches!(
            model
                .forward(&[0.5; 4], &NeuroModulators::default())
                .unwrap_err(),
            HybridError::SnnSpikeOutOfRange {
                neuron: 9,
                num_neurons: 8
            }
        ));
    }

    /// Issue #93 (3): a neuron reported twice in one tick violates the
    /// at-most-once fired-ID contract and is rejected instead of
    /// double-counted.
    #[test]
    fn forward_rejects_duplicate_backend_spikes() {
        let mut model = scripted_router(vec![3, 1, 3], SpikeCountDecoder);
        assert!(matches!(
            model
                .forward_frozen(&[0.5; 4], &NeuroModulators::default())
                .unwrap_err(),
            HybridError::SnnDuplicateSpike { neuron: 3 }
        ));
    }

    /// Issue #93 (3): decoder readouts must be finite and non-negative — the
    /// boost and the hidden-gain normalization both assume activity >= 0.
    #[test]
    fn forward_rejects_negative_or_non_finite_decoder_readout() {
        let mods = NeuroModulators::default();
        for (bad, index) in [(-1.0, 2usize), (f32::INFINITY, 5), (f32::NEG_INFINITY, 0)] {
            let mut readout = vec![0.0f32; 8];
            readout[index] = bad;
            let mut model = scripted_router(Vec::new(), FixedReadoutDecoder(readout));
            match model.forward(&[0.5; 4], &mods).unwrap_err() {
                HybridError::SnnInvalidReadout { index: got, value } => {
                    assert_eq!(got, index);
                    assert_eq!(value, bad);
                }
                other => panic!("expected SnnInvalidReadout, got {other:?}"),
            }
        }
        // NaN keeps the structured non-finite report.
        let mut nan = scripted_router(Vec::new(), FixedReadoutDecoder(vec![f32::NAN; 8]));
        assert!(matches!(
            nan.forward(&[0.5; 4], &mods).unwrap_err(),
            HybridError::NonFinite {
                stage: ForwardValueStage::DecoderReadout,
                index: 0
            }
        ));
    }

    /// A decoder readout of the wrong length is a structural violation, not a
    /// NaN: it reuses the structured length mismatch.
    #[test]
    fn forward_rejects_wrong_length_decoder_readout() {
        let mods = NeuroModulators::default();
        for got in [0usize, 7, 9] {
            let mut model = scripted_router(Vec::new(), FixedReadoutDecoder(vec![0.0; got]));
            assert!(matches!(
                model.forward(&[0.5; 4], &mods).unwrap_err(),
                HybridError::InputLengthMismatch {
                    expected: 8,
                    got: g
                } if g == got
            ));
        }
    }

    /// `spike_boost * activity` saturates instead of overflowing: a finite
    /// gate stays finite and a `-Inf` gate stays `-Inf` rather than becoming
    /// NaN after the backend tick has been consumed.
    #[test]
    fn spike_boost_saturates_instead_of_overflowing() {
        let mods = NeuroModulators::default();
        // Expert 0 gated at -Inf, expert 1 at 0.0; both spike with activity
        // 1e30.
        let mut logits = vec![0.0f32; 8];
        logits[0] = f32::NEG_INFINITY;
        let mut model = SpikingExpertRouter::new(
            ScriptedRouter::new(8, 2, logits),
            ScriptedSnn::new(8, vec![0, 1]),
            RateEncoder::default(),
            ScaledCountDecoder(1e30),
        )
        .unwrap()
        .with_spike_boost(1e30)
        .unwrap();
        let out = model.forward(&[0.5; 4], &mods).unwrap();
        assert_eq!(out.route.selected_experts[0], 1);
        assert!(!out.route.selected_experts.contains(&0));
        assert!(out.route.expert_weights.iter().all(|w| !w.is_nan()));
        assert_eq!(out.route.expert_weights[0], 0.0);
        assert!(out.hidden_gain.is_finite());
    }

    /// A gate already at `f32::MAX` plus a large bonus must saturate rather
    /// than overflow to `+Inf`: the score stays finite and the expert wins.
    /// Expert 1 sits at `f32::MAX` without spiking, so if the spiked expert 0
    /// overflowed to `+Inf` the softmax would hand it all the mass; saturation
    /// leaves both at `f32::MAX` and they share it.
    #[test]
    fn finite_gate_plus_large_bonus_saturates_to_finite_score() {
        let mods = NeuroModulators::default();
        let mut logits = vec![0.0f32; 8];
        logits[0] = f32::MAX;
        logits[1] = f32::MAX;
        // Expert 0 spikes with activity 1e30 under a 1e30 boost.
        let mut model = SpikingExpertRouter::new(
            ScriptedRouter::new(8, 2, logits),
            ScriptedSnn::new(8, vec![0]),
            RateEncoder::default(),
            ScaledCountDecoder(1e30),
        )
        .unwrap()
        .with_spike_boost(1e30)
        .unwrap();
        let out = model.forward(&[0.5; 4], &mods).unwrap();
        assert_eq!(out.route.selected_experts, vec![0, 1]);
        assert!(out.route.expert_weights.iter().all(|w| w.is_finite()));
        assert_eq!(
            out.route.expert_weights[0], out.route.expert_weights[1],
            "both scores saturate at f32::MAX, so neither may absorb the mass"
        );
        assert!(out.route.expert_weights[0] > 0.0);
        assert!(out.hidden_gain.is_finite());
    }

    /// Spikes only on unselected experts must not zero the hidden state: the
    /// gain falls back to the dense selected mass exactly as on a silent tick.
    #[test]
    fn hidden_keeps_dense_mass_when_only_unselected_experts_spike() {
        let mods = NeuroModulators::default();
        // Expert 7 spikes but -2.75 + 1.0 still loses to -1.0 / -1.25.
        let mut model = scripted_router(vec![7], SpikeCountDecoder);
        let out = model.forward(&[0.5; 4], &mods).unwrap();
        assert_eq!(out.route.selected_experts, vec![0, 1]);
        let dense_mass: f32 = out
            .route
            .selected_experts
            .iter()
            .map(|&e| out.route.expert_weights[e])
            .sum();
        assert!(dense_mass > 0.0);
        assert_eq!(out.hidden_gain, dense_mass);
    }

    /// Neurons beyond the expert set are excluded from routing and the
    /// hidden gain, still validated, and still reported in `fired_neurons`.
    #[test]
    fn extra_neurons_do_not_affect_routing_but_are_reported() {
        let mods = NeuroModulators::default();
        let mut model = SpikingExpertRouter::new(
            ScriptedRouter::new(8, 2, NEGATIVE_LOGITS.to_vec()),
            ScriptedSnn::new(12, vec![11]),
            RateEncoder::default(),
            SpikeCountDecoder,
        )
        .unwrap();
        let out = model.forward(&[0.5; 4], &mods).unwrap();
        assert_eq!(out.num_neurons, 12);
        // Spike on neuron 11 did not promote any expert: dense ranking.
        assert_eq!(out.route.selected_experts, vec![0, 1]);
        assert_eq!(out.fired_neurons, vec![11]);
        // Readout index 11 is not an expert, so the gain is the dense mass.
        let dense_mass: f32 = out
            .route
            .selected_experts
            .iter()
            .map(|&e| out.route.expert_weights[e])
            .sum();
        assert_eq!(out.hidden_gain, dense_mass);
    }

    /// Non-finite modulators are caller-side garbage; reject them before the
    /// encoder advances, mirroring the pre-tick ordering.
    #[test]
    fn non_finite_modulators_are_rejected_before_the_encoder_advances() {
        let mut model = SpikingExpertRouter::new(
            ScriptedRouter::new(8, 2, NEGATIVE_LOGITS.to_vec()),
            MockSnn::new(8, 8, 0.0),
            CountingEncoder::new(1.0),
            SpikeCountDecoder,
        )
        .unwrap();
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mods = NeuroModulators {
                dopamine: bad,
                ..NeuroModulators::default()
            };
            match model.forward(&[0.5; 4], &mods).unwrap_err() {
                HybridError::InvalidConfig(msg) => {
                    assert!(msg.contains("dopamine"), "{msg}")
                }
                other => panic!("unexpected {other:?}"),
            }
            assert_eq!(model.encoder().calls, 0, "encoder advanced");
            assert_eq!(model.snn().steps, 0);
        }
    }

    /// A backend reset failure propagates and leaves the encoder epoch
    /// untouched; a successful reset resets both.
    #[test]
    fn reset_propagates_and_only_clears_encoder_after_snn_resets() {
        let mut failing = SpikingExpertRouter::new(
            ScriptedRouter::new(8, 2, NEGATIVE_LOGITS.to_vec()),
            ResetFailsSnn {
                channels: 8,
                neurons: 8,
            },
            CountingEncoder::new(1.0),
            SpikeCountDecoder,
        )
        .unwrap();
        assert!(failing.reset().is_err());
        assert_eq!(failing.encoder().resets, 0, "encoder reset must wait");

        let mut ok = spiking_router(MockSnn::new(8, 8, 0.0));
        ok.reset().unwrap();
        assert_eq!(ok.snn().steps, 0);
    }
}
