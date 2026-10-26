// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-crate SNN step-contract parity tests (Linear RM-2360, ADR 0002).
//!
//! cortex-tensor keeps its own `src/snn` contract (`SnnBackend`,
//! `SnnEncoder`/`SnnDecoder`, `SpikingMoeRouter`, `NeuromodNetwork`) because
//! the crate is deliberately ANN+SNN capable (decision RM-2206/RM-2360; the
//! removal PRs rmems/cortex-tensor#137 and #141 were closed unmerged).
//! hybrid-fusion owns a twin contract (`SpikingNetwork`, `SpikeEncoder`/
//! `SpikeDecoder`, `SpikingExpertRouter`, `NeuromodSnn`) for its orchestrator
//! role. The two are held together by a documented mapping —
//! `docs/adr/0002-snn-step-contract-parity.md` in this repo, mirrored in
//! cortex-tensor as `docs/snn-contract-mapping.md` — and by these tests,
//! which run the same seeded inputs through both trait sets and fail when
//! they drift apart.
//!
//! # What must agree (asserted)
//!
//! * Seeded identity-wired neuromod adapters replay bit-for-bit: both wrap
//!   `neuromod::SpikingNetwork::with_dimensions(n, 1, n)` with the diagonal
//!   at `w_max`, drive `step`/`step_frozen` through `step_with_rng` with the
//!   same `StdRng` seed, and receive the same effective neuromodulator frame.
//! * The spiking routers produce the same boosted selection from the same
//!   gate logits, spike boost (`DEFAULT_SPIKE_BOOST == 1.0` in both crates),
//!   and readouts, on learning and held-out ticks alike.
//! * Host-side validation: a backend firing an out-of-range or duplicate ID
//!   is rejected by both routers before the readout can influence routing.
//! * The #91 never-spiking-channel fix exists on both sides. The tests
//!   assert cortex-tensor's public rejection (`from_network` on a blank
//!   network); hybrid-fusion's copy sits behind its `pub(crate)` boundary
//!   and its public constructors cannot produce a dead network, so its side
//!   is covered by that crate's in-crate tests, not by this gate.
//!
//! # Intentional differences (documented in ADR 0002, asserted here)
//!
//! * Width axes: hybrid-fusion separates `num_channels` (stimulus width) from
//!   `num_neurons` (output population); cortex-tensor requires one LIF neuron
//!   per channel, so `NeuromodNetwork::new(8, 1, 4)` is an error while
//!   `NeuromodSnn::with_seed(4, 8, ..)` is fine.
//! * `SpikeCountDecoder`: hybrid-fusion's decoder is fallible, sized by the
//!   output population, and rejects out-of-range fired IDs; cortex-tensor's
//!   is infallible, sized by the channel count, and silently drops
//!   out-of-range IDs because its *host* (`SpikingMoeRouter`) validates the
//!   spike list before decoding. Both routers fail closed; only the
//!   standalone-decoder behavior differs.
//! * Reset semantics: cortex-tensor's `reset` rewinds the replay RNG to the
//!   seed and keeps learned weights; hybrid-fusion's `reset` restores the
//!   construction-time weights/thresholds and keeps the RNG stream (explicit
//!   `reseed` restarts it). Both replay recipes are pinned below.
//! * Output shape: `SpikingMoeRouter` returns the scaled hidden state
//!   (`embedding × gain`); `SpikingExpertRouter` reports the scalar
//!   `hidden_gain` and the fired IDs instead. The parity assertion is
//!   `cortex.hidden ≈ embedding × hybrid.hidden_gain`.
//! * Capabilities vocabulary: cortex-tensor's `SnnCapabilities` struct maps
//!   onto hybrid-fusion's `BackendCapabilities` + `RequiredFeature` strings
//!   (asserted field by field below).
//!
//! # Fixtures
//!
//! The adapters use all-zero neuromodulators: hybrid-fusion's converter maps
//! an all-zero `NeuroModulators` to an all-zero `neuromod::NeuroModulators`,
//! which is exactly the `NeuroModulators::default()` frame cortex-tensor's
//! adapter applies to every step, so both backends receive the identical
//! effective frame without this package re-stating hybrid-fusion's field
//! mapping.
//!
//! The routers use each crate's own synthetic gate policy: cortex-tensor's
//! stub `MoeRouter` (empty path, `DenseSim`) and hybrid-fusion's
//! `synthetic_gate_scores`. The two chunkings of the embedding coincide
//! exactly when `num_experts` divides the embedding width, which the
//! precondition in the router test pins (`2048 % 8 == 0`), so both routers
//! see identical gate logits bit-for-bit.
//!
//! Float tolerance: `selected_experts` and fired-ID lists are compared
//! exactly (identical `f32` inputs and operation order on both sides), while
//! `expert_weights` and the hidden state use a small tolerance because the
//! two crates' softmax kernels differ in accumulation detail (cortex-tensor
//! exponentiates in `f64` and applies a simplex residual; hybrid-fusion
//! exponentiates in `f32` and accumulates the normalizer in `f64`).

use cortex_tensor::CortexError;
use cortex_tensor::moe::MoeRouter;
use cortex_tensor::snn::{
    DEFAULT_SPIKE_BOOST, NeuromodNetwork, RateEncoder as CortexRateEncoder, SnnBackend,
    SnnCapabilities, SnnDecoder as CortexSnnDecoder, SnnEncoder as CortexSnnEncoder, SnnStepOutput,
    SpikeCountDecoder as CortexSpikeCountDecoder, SpikingMoeRouter,
};
use cortex_tensor::types::EMBEDDING_DIM;
use hybrid_fusion::{
    BackendCapabilities, DEFAULT_SPIKE_BOOST as HYBRID_SPIKE_BOOST, ExpertRouteOutput,
    ExpertRouter, HybridError, NeuroModulators, NeuromodSnn, RateEncoder as HybridRateEncoder,
    Result as HybridResult, SpikeCountDecoder as HybridSpikeCountDecoder,
    SpikeDecoder as HybridSpikeDecoder, SpikeEncoder as HybridSpikeEncoder, SpikingExpertRouter,
    SpikingNetwork, routing_entropy, softmax, synthetic_gate_scores, top_k_indices,
};
use neuromod::SpikingNetwork as NeuromodEngine;

/// Output population / expert count used by the router fixtures. Must divide
/// [`EMBEDDING_DIM`] so the two crates' synthetic gate chunkings coincide.
const EXPERTS: usize = 8;
const TOP_K: usize = 4;
/// Backend width for the adapter fixtures.
const CHANNELS: usize = 8;
const STEPS: usize = 6;
const SEED: u64 = 0x5EED_2360;
/// One short of [`CHANNELS`]; a named const so it can appear in a pattern.
const SHORT: usize = CHANNELS - 1;

/// hybrid-fusion modulator frame that converts to the all-zero
/// `neuromod::NeuroModulators` cortex-tensor's adapter uses by default.
fn zero_modulators() -> NeuroModulators {
    NeuroModulators {
        dopamine: 0.0,
        cortisol: 0.0,
        acetylcholine: 0.0,
        tempo: 1.0,
        aux_dopamine: 0.0,
    }
}

/// Per-tick stimulus for the adapter fixtures: channel `i` gets a magnitude
/// in `0.1..=2.2` (rotated by tick), so some channels stay quiet, some fire
/// reliably, and the pattern varies per tick.
fn stimulus_for_tick(tick: usize) -> Vec<f32> {
    (0..CHANNELS)
        .map(|ch| 0.1 + 0.3 * (((ch + tick) % 8) as f32))
        .collect()
}

/// cortex-tensor `SpikingMoeRouter::forward`/`forward_frozen` embeddings:
/// `[EMBEDDING_DIM]` values whose per-expert chunk sums reproduce the
/// [`stimulus_for_tick`] magnitudes, so the router fixture and the adapter
/// fixture drive comparable stimuli into the same backends.
fn embedding_for_tick(tick: usize) -> Vec<f32> {
    let chunk = EMBEDDING_DIM / EXPERTS;
    (0..EMBEDDING_DIM)
        .map(|i| (0.1 + 0.3 * (((i / chunk + tick) % 8) as f32)) / chunk as f32)
        .collect()
}

/// hybrid-fusion `ExpertRouter` that mirrors cortex-tensor's stub
/// `MoeRouter` gate policy: the synthetic chunk sums of
/// [`synthetic_gate_scores`]. Requires `EMBEDDING_DIM % num_experts == 0` so
/// both chunkings partition the embedding identically.
struct SyntheticChunkRouter {
    num_experts: usize,
    top_k: usize,
}

impl ExpertRouter for SyntheticChunkRouter {
    fn num_experts(&self) -> usize {
        self.num_experts
    }

    fn top_k(&self) -> usize {
        self.top_k
    }

    fn route(&mut self, embedding: &[f32]) -> HybridResult<ExpertRouteOutput> {
        let logits = self.gate_logits(embedding)?;
        let expert_weights = softmax(&logits)?;
        let selected_experts = top_k_indices(&logits, self.top_k);
        let entropy = routing_entropy(&expert_weights);
        Ok(ExpertRouteOutput {
            expert_weights,
            selected_experts,
            routing_entropy: Some(entropy),
        })
    }

    fn gate_logits(&mut self, embedding: &[f32]) -> HybridResult<Vec<f32>> {
        if embedding.is_empty() {
            return Err(HybridError::InvalidConfig(
                "SyntheticChunkRouter: embedding must be non-empty".into(),
            ));
        }
        Ok(synthetic_gate_scores(self.num_experts, embedding))
    }
}

fn assert_close(actual: f32, expected: f32, what: &str) {
    assert!(
        (actual - expected).abs() <= 1e-5,
        "{what}: {actual} vs {expected} (|diff| > 1e-5)"
    );
}

fn assert_fired_envelope(fired: &[Vec<usize>]) {
    let total: usize = fired.iter().map(Vec::len).sum();
    assert!(
        (1..=CHANNELS * STEPS).contains(&total),
        "spike total {total} outside envelope 1..={}",
        CHANNELS * STEPS
    );
    for (tick, indices) in fired.iter().enumerate() {
        assert!(
            indices.iter().all(|&idx| idx < CHANNELS),
            "tick {tick} fired outside 0..{CHANNELS}: {indices:?}"
        );
    }
}

fn hybrid_advertises(caps: &BackendCapabilities, feature: &str) -> bool {
    caps.features.iter().any(|f| f.as_str() == feature)
}

fn assert_adapter_capabilities_match(cortex: &NeuromodNetwork, hybrid: &NeuromodSnn) {
    // SnnCapabilities (cortex-tensor) ↔ BackendCapabilities + features
    // (hybrid-fusion): same backend identity and the same five advertised
    // facts, in two vocabularies.
    let caps = cortex.capabilities();
    let hcaps = hybrid.capabilities();
    assert_eq!(caps.backend_name, hcaps.backend_name);
    assert_eq!(caps.backend_name, "neuromod::SpikingNetwork");
    assert_eq!(
        caps.frozen_evaluation,
        hybrid_advertises(&hcaps, "frozen_evaluation")
    );
    assert_eq!(caps.plasticity, hybrid_advertises(&hcaps, "plasticity"));
    assert_eq!(caps.caller_rng, hybrid_advertises(&hcaps, "caller_rng"));
    assert_eq!(
        caps.neuromodulation,
        hybrid_advertises(&hcaps, "neuromodulation")
    );
    assert!(hybrid_advertises(&hcaps, "step"));
    assert_eq!(hybrid.num_channels(), CHANNELS);
    assert_eq!(hybrid.num_neurons(), CHANNELS);
    assert!(
        caps.caller_rng,
        "cortex adapter reports caller_rng once seeded"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Boundary adapters: encoder/decoder defaults, decoder difference, host checks
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn rate_encoder_and_spike_boost_defaults_match() {
    // DEFAULT_SPIKE_BOOST is part of the shared routing contract: both crates
    // define the additive boost per unit of decoded activity as 1.0.
    assert_eq!(HYBRID_SPIKE_BOOST, DEFAULT_SPIKE_BOOST);

    let signal = [0.25_f32, -0.5, 1.0];
    let cortex = CortexRateEncoder::default()
        .encode(&signal, 6)
        .expect("cortex encode");
    let hybrid = HybridRateEncoder::default()
        .encode(&signal, 6)
        .expect("hybrid encode");
    assert_eq!(cortex, hybrid, "RateEncoder tiling differs across crates");
    assert_eq!(cortex.len(), 6);

    // A stateless encoder reset is a no-op on both sides.
    let mut cortex_enc = CortexRateEncoder::default();
    let mut hybrid_enc = HybridRateEncoder::default();
    cortex_enc.reset();
    hybrid_enc.reset();
    assert_eq!(
        cortex_enc.encode(&signal, 6).expect("cortex encode"),
        hybrid_enc.encode(&signal, 6).expect("hybrid encode")
    );
}

#[test]
fn spike_count_decoders_documented_difference_and_host_parity() {
    // Intentional difference (ADR 0002): hybrid-fusion's decoder fails closed
    // on an out-of-range fired ID …
    let err = HybridSpikeCountDecoder
        .decode(&[99], 4)
        .expect_err("hybrid decoder must reject out-of-range fired IDs");
    assert!(matches!(
        err,
        HybridError::SnnSpikeOutOfRange {
            neuron: 99,
            num_neurons: 4
        }
    ));
    // … while cortex-tensor's silently drops it: its `SpikingMoeRouter` host
    // validates the spike list before decoding, so a compliant flow never
    // reaches the decoder with a bad ID.
    let counts = CortexSpikeCountDecoder.decode(
        &SnnStepOutput {
            spikes: vec![99, 1],
        },
        4,
    );
    assert_eq!(counts, vec![0.0, 1.0, 0.0, 0.0]);

    // Host parity: both routers reject a misbehaving backend (out-of-range
    // and duplicate fired IDs) before the readout can influence routing.
    struct MisfiringCortex {
        spikes: Vec<usize>,
    }
    impl SnnBackend for MisfiringCortex {
        fn channels(&self) -> usize {
            CHANNELS
        }
        fn step(&mut self, _stimulus: &[f32]) -> cortex_tensor::Result<SnnStepOutput> {
            Ok(SnnStepOutput {
                spikes: self.spikes.clone(),
            })
        }
        fn step_frozen(&mut self, stimulus: &[f32]) -> cortex_tensor::Result<SnnStepOutput> {
            self.step(stimulus)
        }
        fn reset(&mut self) {}
        fn capabilities(&self) -> SnnCapabilities {
            SnnCapabilities {
                backend_name: "misfiring-stub",
                frozen_evaluation: true,
                plasticity: false,
                caller_rng: false,
                neuromodulation: false,
            }
        }
    }

    struct MisfiringHybrid {
        fired: Vec<usize>,
    }
    impl SpikingNetwork for MisfiringHybrid {
        fn step(
            &mut self,
            _stimuli: &[f32],
            _modulators: &NeuroModulators,
        ) -> HybridResult<Vec<usize>> {
            Ok(self.fired.clone())
        }
        fn num_channels(&self) -> usize {
            CHANNELS
        }
    }

    let embedding = embedding_for_tick(0);
    for (fired, duplicate) in [(&[CHANNELS][..], false), (&[2usize, 2][..], true)] {
        let mut cortex_router = SpikingMoeRouter::new(
            MoeRouter::load("", EXPERTS, TOP_K).expect("stub MoeRouter"),
            MisfiringCortex {
                spikes: fired.to_vec(),
            },
            CortexRateEncoder::default(),
            CortexSpikeCountDecoder,
        )
        .expect("cortex spiking router");
        let err = cortex_router
            .forward(&embedding)
            .expect_err("cortex host must reject the misfiring backend");

        let mut hybrid_router = SpikingExpertRouter::new(
            SyntheticChunkRouter {
                num_experts: EXPERTS,
                top_k: TOP_K,
            },
            MisfiringHybrid {
                fired: fired.to_vec(),
            },
            HybridRateEncoder::default(),
            HybridSpikeCountDecoder,
        )
        .expect("hybrid spiking router");
        let h_err = hybrid_router
            .forward(&embedding, &zero_modulators())
            .expect_err("hybrid host must reject the misfiring backend");

        if duplicate {
            assert!(matches!(err, CortexError::SnnDuplicateSpike { channel: 2 }));
            assert!(matches!(
                h_err,
                HybridError::SnnDuplicateSpike { neuron: 2 }
            ));
        } else {
            assert!(matches!(
                err,
                CortexError::SnnSpikeOutOfRange {
                    channel: CHANNELS,
                    channels: CHANNELS
                }
            ));
            assert!(matches!(
                h_err,
                HybridError::SnnSpikeOutOfRange {
                    neuron: CHANNELS,
                    num_neurons: CHANNELS
                }
            ));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Neuromod adapters: seeded stepping must replay identically
// ─────────────────────────────────────────────────────────────────────────

fn cortex_adapter() -> NeuromodNetwork {
    NeuromodNetwork::new(CHANNELS, 1, CHANNELS)
        .expect("cortex NeuromodNetwork")
        .with_seed(SEED)
}

fn hybrid_adapter() -> NeuromodSnn {
    NeuromodSnn::identity_wired(CHANNELS, SEED).expect("hybrid NeuromodSnn")
}

#[test]
fn identity_wired_adapters_step_identically() {
    // Both adapters must be drivable at all (the #91 fix: no channel that
    // can never spike) and must agree bit-for-bit on every learning tick.
    let mut cortex = cortex_adapter();
    let mut hybrid = hybrid_adapter();
    assert_adapter_capabilities_match(&cortex, &hybrid);

    let mut all_hybrid = Vec::new();
    for tick in 0..STEPS {
        let stimulus = stimulus_for_tick(tick);
        let c = SnnBackend::step(&mut cortex, &stimulus).expect("cortex step");
        let h =
            SpikingNetwork::step(&mut hybrid, &stimulus, &zero_modulators()).expect("hybrid step");
        assert_eq!(c.spikes, h, "learning tick {tick} diverged across crates");
        all_hybrid.push(h);
    }
    assert_fired_envelope(&all_hybrid);

    // Both adapters reject a wrong-width stimulus before consuming a tick.
    let short = vec![0.5_f32; SHORT];
    assert!(SnnBackend::step(&mut cortex, &short).is_err());
    let err = SpikingNetwork::step(&mut hybrid, &short, &zero_modulators())
        .expect_err("hybrid adapter must reject a short stimulus");
    assert!(matches!(
        err,
        HybridError::InputLengthMismatch {
            expected: CHANNELS,
            got: SHORT
        }
    ));
}

#[test]
fn frozen_steps_and_reset_recipes_replay_identically() {
    let mut cortex = cortex_adapter();
    let mut hybrid = hybrid_adapter();
    let mut control = hybrid_adapter();
    let stimuli: Vec<Vec<f32>> = (0..STEPS).map(stimulus_for_tick).collect();

    // Frozen ticks agree bit-for-bit, like learning ticks.
    let mut first_cortex = Vec::new();
    let mut first_hybrid = Vec::new();
    for (tick, stimulus) in stimuli.iter().enumerate() {
        let c = SnnBackend::step_frozen(&mut cortex, stimulus).expect("cortex frozen step");
        let h = SpikingNetwork::step_frozen(&mut hybrid, stimulus, &zero_modulators())
            .expect("hybrid frozen step");
        assert_eq!(c.spikes, h, "frozen tick {tick} diverged across crates");
        first_cortex.push(c.spikes);
        first_hybrid.push(h);
    }
    assert_fired_envelope(&first_hybrid);

    // Control run on a second hybrid instance: same seed, same sequence …
    let control_first: Vec<Vec<usize>> = stimuli
        .iter()
        .map(|s| {
            SpikingNetwork::step_frozen(&mut control, s, &zero_modulators())
                .expect("control frozen step")
        })
        .collect();
    assert_eq!(
        first_hybrid, control_first,
        "same-seed hybrid instances diverged"
    );
    // … then the sequence again with no reset: the RNG stream continues.
    let control_second: Vec<Vec<usize>> = stimuli
        .iter()
        .map(|s| {
            SpikingNetwork::step_frozen(&mut control, s, &zero_modulators())
                .expect("control frozen step")
        })
        .collect();

    // cortex-tensor reset recipe: `reset` rewinds the RNG to the seed, so the
    // frozen sequence replays (frozen stepping leaves weights untouched).
    SnnBackend::reset(&mut cortex);
    let cortex_replay: Vec<Vec<usize>> = stimuli
        .iter()
        .map(|s| {
            SnnBackend::step_frozen(&mut cortex, s)
                .expect("cortex frozen step")
                .spikes
        })
        .collect();
    assert_eq!(
        first_cortex, cortex_replay,
        "cortex reset() must rewind the replay stream"
    );

    // Intentional difference (ADR 0002): hybrid-fusion's `reset` restores the
    // construction-time weights/thresholds but does NOT reseed the RNG, so
    // the sequence after a bare reset is bit-for-bit the *continuation* a
    // never-reset instance produced, not a replay of the first sequence.
    hybrid.reset();
    let hybrid_continued: Vec<Vec<usize>> = stimuli
        .iter()
        .map(|s| {
            SpikingNetwork::step_frozen(&mut hybrid, s, &zero_modulators())
                .expect("hybrid frozen step")
        })
        .collect();
    assert_eq!(
        hybrid_continued, control_second,
        "hybrid reset() must continue the RNG stream, not rewind it"
    );

    // hybrid-fusion replay recipe: `reset` + `reseed(seed)`.
    hybrid.reset();
    hybrid.reseed(SEED);
    let hybrid_replay: Vec<Vec<usize>> = stimuli
        .iter()
        .map(|s| {
            SpikingNetwork::step_frozen(&mut hybrid, s, &zero_modulators())
                .expect("hybrid frozen step")
        })
        .collect();
    assert_eq!(
        first_hybrid, hybrid_replay,
        "hybrid reset() + reseed(seed) must replay"
    );
}

#[test]
fn width_rules_and_never_spiking_rejection_match_the_mapping() {
    // Intentional difference (ADR 0002): hybrid-fusion separates the output
    // population from the stimulus width; cortex-tensor requires one LIF
    // neuron per channel because `channels()` is its whole width contract.
    let wide = NeuromodSnn::with_seed(4, 8, SEED).expect("population != width is allowed");
    assert_eq!(wide.num_channels(), 4);
    assert_eq!(wide.num_neurons(), 8);
    // `NeuromodNetwork` is not `Debug`, so spell the unwrap-or-panic out.
    let err = match NeuromodNetwork::new(8, 1, 4) {
        Ok(_) => panic!("one LIF per channel is required"),
        Err(err) => err,
    };
    assert!(
        matches!(err, CortexError::InvalidConfig(_)),
        "unexpected error: {err:?}"
    );

    // #91 parity: both adapters reject a network whose LIF neurons can never
    // spike. cortex-tensor exposes the check through the public
    // `from_network`; hybrid-fusion's equivalent lives behind its
    // `pub(crate)` `from_network` boundary (no `neuromod` type may cross its
    // public API), and its public constructors seed strictly positive
    // weights so the rejection cannot trigger through them.
    let blank = NeuromodEngine::with_dimensions(4, 1, 4);
    let err = match NeuromodNetwork::from_network(blank) {
        Ok(_) => panic!("blank network must be rejected (never-spiking channel)"),
        Err(err) => err,
    };
    assert!(
        matches!(err, CortexError::InvalidConfig(_)),
        "unexpected error: {err:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Spiking routers: same seeded inputs, same boosted route
// ─────────────────────────────────────────────────────────────────────────

fn assert_routes_agree(
    cortex: &cortex_tensor::moe::MoeOutput,
    hybrid: &hybrid_fusion::SpikingRouteOutput,
    embedding: &[f32],
    tick: usize,
) {
    assert_eq!(
        cortex.selected_experts, hybrid.route.selected_experts,
        "tick {tick}: boosted selection diverged across crates"
    );
    for (expert, (&c, &h)) in cortex
        .expert_weights
        .iter()
        .zip(hybrid.route.expert_weights.iter())
        .enumerate()
    {
        assert_close(c, h, &format!("tick {tick}: expert {expert} weight"));
    }
    // Intentional output-shape difference (ADR 0002): cortex-tensor returns
    // the scaled hidden state, hybrid-fusion the scalar gain — same number.
    assert!(
        (0.0..=1.0).contains(&hybrid.hidden_gain),
        "tick {tick}: hidden_gain {} outside [0, 1]",
        hybrid.hidden_gain
    );
    for (i, (&hidden, &value)) in cortex.hidden.iter().zip(embedding.iter()).enumerate() {
        // Embedding values are tiny (about 4e-4..9e-3), so an absolute
        // tolerance on the scaled hidden state could not see gain drift:
        // compare the implied gain instead. Every fixture value is strictly
        // positive, so the division is exact enough for the 1e-5 tolerance.
        assert_close(
            hidden / value,
            hybrid.hidden_gain,
            &format!("tick {tick}: implied hidden gain at [{i}]"),
        );
    }
}

fn cortex_spiking_router()
-> SpikingMoeRouter<NeuromodNetwork, CortexRateEncoder, CortexSpikeCountDecoder> {
    SpikingMoeRouter::new(
        MoeRouter::load("", EXPERTS, TOP_K).expect("stub MoeRouter"),
        cortex_adapter(),
        CortexRateEncoder::default(),
        CortexSpikeCountDecoder,
    )
    .expect("cortex SpikingMoeRouter")
}

type HybridSpikingRouter = SpikingExpertRouter<
    SyntheticChunkRouter,
    NeuromodSnn,
    HybridRateEncoder,
    HybridSpikeCountDecoder,
>;

fn hybrid_spiking_router() -> HybridSpikingRouter {
    SpikingExpertRouter::new(
        SyntheticChunkRouter {
            num_experts: EXPERTS,
            top_k: TOP_K,
        },
        hybrid_adapter(),
        HybridRateEncoder::default(),
        HybridSpikeCountDecoder,
    )
    .expect("hybrid SpikingExpertRouter")
}

#[test]
fn spiking_routers_agree_on_seeded_learning_ticks() {
    assert_eq!(
        EMBEDDING_DIM % EXPERTS,
        0,
        "fixture requires divisible width"
    );
    let mut cortex = cortex_spiking_router();
    let mut hybrid = hybrid_spiking_router();

    let mut fired_total = 0usize;
    for tick in 0..STEPS {
        let embedding = embedding_for_tick(tick);
        let c = cortex.forward(&embedding).expect("cortex forward");
        let h = hybrid
            .forward(&embedding, &zero_modulators())
            .expect("hybrid forward");
        fired_total += h.fired_neurons.len();
        assert_routes_agree(&c, &h, &embedding, tick);
    }
    assert!(fired_total > 0, "fixture never spiked; parity is vacuous");
}

#[test]
fn spiking_routers_agree_on_seeded_frozen_ticks() {
    let mut cortex = cortex_spiking_router();
    let mut hybrid = hybrid_spiking_router();

    let mut fired_total = 0usize;
    for tick in 0..STEPS {
        let embedding = embedding_for_tick(tick);
        // Both crates expose the held-out path; hybrid-fusion additionally
        // gates it on the frozen_evaluation capability (asserted in the
        // adapter test).
        let c = cortex
            .forward_frozen(&embedding)
            .expect("cortex forward_frozen");
        let h = hybrid
            .forward_frozen(&embedding, &zero_modulators())
            .expect("hybrid forward_frozen");
        fired_total += h.fired_neurons.len();
        assert_routes_agree(&c, &h, &embedding, tick);
    }
    assert!(fired_total > 0, "fixture never spiked; parity is vacuous");
}
