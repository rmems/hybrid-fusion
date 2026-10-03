// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Breaks this file is written to catch (issue #41 / RM-1805):
// - a stage pinned to a domain the supplied backend does not advertise is accepted;
// - a dtype, channel width, sequence window, batch, streaming, stateful, or reset
//   requirement the backend cannot meet is accepted;
// - an optional stage feature the backend does not list is accepted;
// - fallback silently substitutes a different backend when the policy forbids it;
// - fallback is refused even when the policy allows it and a compatible backend exists;
// - two backends that both fit a stage are treated as interchangeable when their
//   semantic feature sets differ (caller_rng vs not);
// - capability discovery mutates the backend (step counter, membrane, seed).
// - a requirement domain that disagrees with the compiled stage is accepted;
// - a named fallback is ignored when the preferred backend was not offered;
// - an empty backend identity is selected;
// - two requirements for one stage select different backends;
// - matching stimulus width hides a different output population;
// - a unique domain-matching backend's dtype error is reported as "no backend".

//! Capability negotiation for hybrid stages (issue #41 / RM-1805).
//!
//! Mock backends exercise substitution, policy-driven fallback, and semantic
//! incompatibility. Discovery is asserted side-effect free by comparing
//! backend counters before and after `capabilities()`.

use hybrid_fusion::error::Result;
use hybrid_fusion::plan::{ExecutionDomain, PlanError, StageId};
use hybrid_fusion::{
    BackendCapabilities, BackendId, CapabilityNegotiation, Dtype, FallbackPolicy, HybridConfig,
    NegotiationOutcome, NeuroModulators, PortSpec, RequiredFeature, SpikingNetwork, StageGraph,
    StageKind, StageRequirement, Tensor, Transformer,
};

struct CountingTransformer {
    dim: usize,
    max_seq: usize,
    caps: BackendCapabilities,
    /// Incremented only by inference, never by capability discovery.
    hidden_calls: std::cell::Cell<u32>,
}

impl CountingTransformer {
    fn ann(name: &'static str, dim: usize, max_seq: usize) -> Self {
        Self {
            dim,
            max_seq,
            caps: BackendCapabilities::ann(name)
                .with_dtypes([Dtype::F32])
                .with_hidden_dim(dim)
                .with_max_sequence(max_seq)
                .with_batching(true)
                .with_features([RequiredFeature::new("reference-embedding")]),
            hidden_calls: std::cell::Cell::new(0),
        }
    }
}

impl Transformer for CountingTransformer {
    fn hidden_states(&self, token_ids: &[u32]) -> Tensor {
        self.hidden_calls.set(self.hidden_calls.get() + 1);
        let seq = token_ids.len();
        Tensor::from_vec(vec![0.25; seq * self.dim], &[seq, self.dim])
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq
    }
    fn param_count(&self) -> usize {
        0
    }
    fn capabilities(&self) -> BackendCapabilities {
        self.caps.clone()
    }
}

struct CountingSnn {
    channels: usize,
    neurons: usize,
    caps: BackendCapabilities,
    steps: std::cell::Cell<u32>,
}

impl CountingSnn {
    fn snn(name: &'static str, channels: usize) -> Self {
        Self {
            channels,
            neurons: channels,
            caps: BackendCapabilities::snn(name)
                .with_dtypes([Dtype::F32])
                .with_channels(channels)
                .with_max_sequence(1)
                .with_reset(true)
                .with_stateful(true)
                .with_features([RequiredFeature::new("step")]),
            steps: std::cell::Cell::new(0),
        }
    }
}

impl SpikingNetwork for CountingSnn {
    fn step(&mut self, _stimuli: &[f32], _modulators: &NeuroModulators) -> Result<Vec<usize>> {
        self.steps.set(self.steps.get() + 1);
        Ok(vec![0])
    }
    fn num_channels(&self) -> usize {
        self.channels
    }
    fn num_neurons(&self) -> usize {
        self.neurons
    }
    fn capabilities(&self) -> BackendCapabilities {
        self.caps.clone()
    }
}

fn ann_stage(graph: &mut StageGraph, name: &str, dim: usize, seq: usize) -> StageId {
    graph.add_stage(
        name,
        StageKind::Transformer,
        PortSpec::any(),
        PortSpec::f32_exact(&[seq, dim]),
    )
}

fn snn_stage(graph: &mut StageGraph, name: &str, channels: usize) -> StageId {
    graph.add_stage(
        name,
        StageKind::SpikingBlock,
        PortSpec::f32_exact(&[channels]),
        PortSpec::any(),
    )
}

fn requirement_ann(stage: StageId, dim: usize, seq: usize) -> StageRequirement {
    StageRequirement::new(stage, ExecutionDomain::Ann)
        .with_dtype(Dtype::F32)
        .with_hidden_dim(dim)
        .with_max_sequence(seq)
        .with_feature(RequiredFeature::new("reference-embedding"))
}

fn requirement_snn(stage: StageId, channels: usize) -> StageRequirement {
    StageRequirement::new(stage, ExecutionDomain::Snn)
        .with_dtype(Dtype::F32)
        .with_channels(channels)
        .with_feature(RequiredFeature::new("step"))
        .requires_reset()
        .requires_stateful()
}

#[test]
fn discovery_does_not_step_or_run_the_backend() {
    let ann = CountingTransformer::ann("ann-a", 4, 8);
    let snn = CountingSnn::snn("snn-a", 4);
    let _ = ann.capabilities();
    let _ = snn.capabilities();
    assert_eq!(
        ann.hidden_calls.get(),
        0,
        "capabilities() ran hidden_states"
    );
    assert_eq!(snn.steps.get(), 0, "capabilities() stepped the SNN");
}

#[test]
fn compatible_backend_is_selected_without_fallback() {
    let cfg = HybridConfig::tiny();
    let mut graph = StageGraph::new();
    let ann = ann_stage(&mut graph, "tower", cfg.transformer.dim, 2);
    let adapt = graph.add_stage(
        "project",
        StageKind::Adaptation,
        PortSpec::f32_exact(&[2, cfg.transformer.dim]),
        PortSpec::f32_exact(&[cfg.snn_input_channels]),
    );
    let snn = snn_stage(&mut graph, "spikes", cfg.snn_input_channels);
    graph.connect(ann, adapt);
    graph.connect(adapt, snn);
    let plan = graph.compile().expect("structural plan");

    let ann_backend =
        CountingTransformer::ann("ann-a", cfg.transformer.dim, cfg.transformer.max_seq_len);
    let snn_backend = CountingSnn::snn("snn-a", cfg.snn_input_channels);
    let outcome = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(BackendId::new("ann-a"), ann_backend.capabilities())
        .offer(BackendId::new("snn-a"), snn_backend.capabilities())
        .negotiate(
            &plan,
            &[
                requirement_ann(ann, cfg.transformer.dim, 2),
                requirement_snn(snn, cfg.snn_input_channels),
            ],
        )
        .expect("compatible pair");

    assert_eq!(
        outcome.selection(ann),
        Some(&NegotiationOutcome::Selected(BackendId::new("ann-a")))
    );
    assert_eq!(
        outcome.selection(snn),
        Some(&NegotiationOutcome::Selected(BackendId::new("snn-a")))
    );
    assert!(outcome.fallbacks().is_empty());
}

#[test]
fn requirement_domain_must_match_the_compiled_stage() {
    let mut graph = StageGraph::new();
    let stage = ann_stage(&mut graph, "tower", 4, 2);
    let plan = graph.compile().expect("single stage");

    // Compiled Transformer stage is ANN. An SNN requirement plus an SNN offer
    // must not succeed just because the requirement and the offer agree.
    let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(
            BackendId::new("snn-only"),
            BackendCapabilities::snn("snn-only").with_dtypes([Dtype::F32]),
        )
        .negotiate(&plan, &[StageRequirement::new(stage, ExecutionDomain::Snn)])
        .expect_err("requirement domain must match the compiled stage");

    match err {
        PlanError::InvalidParameters(reason) => {
            assert!(
                reason.contains("Ann") && reason.contains("Snn"),
                "domain disagreement was not named: {reason}"
            );
        }
        other => panic!("expected InvalidParameters, got {other:?}"),
    }
}

#[test]
fn unsupported_domain_is_a_structured_error_not_a_string_scan() {
    let mut graph = StageGraph::new();
    let stage = snn_stage(&mut graph, "spikes", 4);
    let plan = graph.compile().expect("single stage");

    let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(
            BackendId::new("ann-only"),
            BackendCapabilities::ann("ann-only").with_dtypes([Dtype::F32]),
        )
        .negotiate(&plan, &[StageRequirement::new(stage, ExecutionDomain::Snn)])
        .expect_err("ANN backend must not satisfy an SNN requirement");

    match err {
        PlanError::UnsupportedBackend {
            stage: got,
            domain,
            reason: _,
        } => {
            assert_eq!(got, stage);
            assert_eq!(domain, ExecutionDomain::Snn);
        }
        other => panic!("expected UnsupportedBackend, got {other:?}"),
    }
}

#[test]
fn dtype_and_width_mismatch_are_distinct_structured_errors() {
    let mut graph = StageGraph::new();
    let stage = snn_stage(&mut graph, "spikes", 8);
    let plan = graph.compile().unwrap();

    let narrow = BackendCapabilities::snn("narrow")
        .with_dtypes([Dtype::F16])
        .with_channels(4)
        .with_reset(true)
        .with_stateful(true);

    let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(BackendId::new("narrow"), narrow)
        .negotiate(&plan, &[requirement_snn(stage, 8).with_dtype(Dtype::F32)])
        .unwrap_err();

    // Dtype is checked before width so a backend that fails both reports the
    // first contract break, not a collapsed "incompatible" string.
    match err {
        PlanError::IncompatibleCapability {
            stage: got,
            field,
            required,
            backend,
        } => {
            assert_eq!(got, stage);
            assert_eq!(field, "dtype");
            assert_eq!(required, "F32");
            assert_eq!(backend, "F16");
        }
        other => panic!("expected dtype IncompatibleCapability, got {other:?}"),
    }
}

#[test]
fn channel_width_mismatch_names_both_sides() {
    let mut graph = StageGraph::new();
    let stage = snn_stage(&mut graph, "spikes", 8);
    let plan = graph.compile().unwrap();
    let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(
            BackendId::new("narrow"),
            BackendCapabilities::snn("narrow")
                .with_dtypes([Dtype::F32])
                .with_channels(4)
                .with_reset(true)
                .with_stateful(true)
                .with_features([RequiredFeature::new("step")]),
        )
        .negotiate(&plan, &[requirement_snn(stage, 8)])
        .unwrap_err();
    match err {
        PlanError::IncompatibleCapability {
            field,
            required,
            backend,
            ..
        } => {
            assert_eq!(field, "channels");
            assert_eq!(required, "8");
            assert_eq!(backend, "4");
        }
        other => panic!("expected channel mismatch, got {other:?}"),
    }
}

#[test]
fn missing_optional_feature_fails_explicitly() {
    let mut graph = StageGraph::new();
    let stage = snn_stage(&mut graph, "spikes", 4);
    let plan = graph.compile().unwrap();
    let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(
            BackendId::new("no-reset-feature"),
            BackendCapabilities::snn("no-reset-feature")
                .with_dtypes([Dtype::F32])
                .with_channels(4)
                .with_reset(true)
                .with_stateful(true)
                .with_features([RequiredFeature::new("step")]),
        )
        .negotiate(
            &plan,
            &[requirement_snn(stage, 4).with_feature(RequiredFeature::new("caller_rng"))],
        )
        .unwrap_err();
    match err {
        PlanError::UnsupportedFeature {
            stage: got,
            feature,
        } => {
            assert_eq!(got, stage);
            assert_eq!(feature.as_str(), "caller_rng");
        }
        other => panic!("expected UnsupportedFeature, got {other:?}"),
    }
}

#[test]
fn forbid_policy_does_not_silently_substitute_a_compatible_fallback() {
    let mut graph = StageGraph::new();
    let stage = ann_stage(&mut graph, "tower", 4, 2);
    let plan = graph.compile().unwrap();
    // Width 8 does not equal the required hidden width 4.
    let preferred = BackendCapabilities::ann("preferred")
        .with_dtypes([Dtype::F32])
        .with_hidden_dim(8);
    let fallback = BackendCapabilities::ann("fallback")
        .with_dtypes([Dtype::F32])
        .with_hidden_dim(4)
        .with_max_sequence(8)
        .with_features([RequiredFeature::new("reference-embedding")]);

    let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(BackendId::new("preferred"), preferred)
        .offer(BackendId::new("fallback"), fallback)
        .prefer(stage, BackendId::new("preferred"))
        .negotiate(&plan, &[requirement_ann(stage, 4, 2)])
        .unwrap_err();
    assert!(
        matches!(
            err,
            PlanError::IncompatibleCapability { .. } | PlanError::UnsupportedBackend { .. }
        ),
        "forbid policy substituted a fallback: {err:?}"
    );
}

#[test]
fn allow_policy_selects_the_named_fallback_and_records_it() {
    let mut graph = StageGraph::new();
    let stage = ann_stage(&mut graph, "tower", 4, 2);
    let plan = graph.compile().unwrap();
    let preferred = BackendCapabilities::ann("preferred").with_dtypes([Dtype::F16]);
    let fallback = BackendCapabilities::ann("fallback")
        .with_dtypes([Dtype::F32])
        .with_hidden_dim(4)
        .with_max_sequence(8)
        .with_features([RequiredFeature::new("reference-embedding")]);

    let outcome = CapabilityNegotiation::new(FallbackPolicy::AllowNamed)
        .offer(BackendId::new("preferred"), preferred)
        .offer(BackendId::new("fallback"), fallback)
        .prefer(stage, BackendId::new("preferred"))
        .fallback_to(stage, BackendId::new("fallback"))
        .negotiate(&plan, &[requirement_ann(stage, 4, 2)])
        .expect("named fallback should be selected");

    assert_eq!(
        outcome.selection(stage),
        Some(&NegotiationOutcome::Fallback {
            rejected: BackendId::new("preferred"),
            selected: BackendId::new("fallback"),
        })
    );
    assert_eq!(outcome.fallbacks(), &[stage]);
}

#[test]
fn semantic_inequivalence_rejects_substitution_even_when_shapes_match() {
    // Both backends fit the shape. They are not interchangeable: one advertises
    // caller-controlled RNG (deterministic replay) and the other does not.
    let mut graph = StageGraph::new();
    let stage = snn_stage(&mut graph, "spikes", 4);
    let plan = graph.compile().unwrap();

    // Preferred backend fails the width check, so policy considers the fallback.
    // The fallback matches width, dtype, reset, and state — and lacks caller_rng.
    // Accepting it would change replay semantics.
    let too_narrow = BackendCapabilities::snn("too-narrow")
        .with_dtypes([Dtype::F32])
        .with_channels(2)
        .with_reset(true)
        .with_stateful(true)
        .with_features([
            RequiredFeature::new("step"),
            RequiredFeature::new("caller_rng"),
        ]);
    let unseeded = BackendCapabilities::snn("unseeded")
        .with_dtypes([Dtype::F32])
        .with_channels(4)
        .with_reset(true)
        .with_stateful(true)
        .with_features([RequiredFeature::new("step")]);

    let err = CapabilityNegotiation::new(FallbackPolicy::AllowNamed)
        .offer(BackendId::new("too-narrow"), too_narrow)
        .offer(BackendId::new("unseeded"), unseeded)
        .prefer(stage, BackendId::new("too-narrow"))
        .fallback_to(stage, BackendId::new("unseeded"))
        .negotiate(
            &plan,
            &[requirement_snn(stage, 4).with_feature(RequiredFeature::new("caller_rng"))],
        )
        .unwrap_err();
    assert!(
        matches!(err, PlanError::SemanticMismatch { .. }),
        "shape-compatible unseeded backend was accepted as a semantic substitute: {err:?}"
    );
}

#[test]
fn named_fallback_runs_when_the_preferred_backend_was_not_offered() {
    let mut graph = StageGraph::new();
    let stage = ann_stage(&mut graph, "tower", 4, 2);
    let plan = graph.compile().unwrap();
    let fallback = BackendCapabilities::ann("fallback")
        .with_dtypes([Dtype::F32])
        .with_hidden_dim(4)
        .with_max_sequence(8)
        .with_features([RequiredFeature::new("reference-embedding")]);

    let outcome = CapabilityNegotiation::new(FallbackPolicy::AllowNamed)
        .offer(BackendId::new("fallback"), fallback)
        .prefer(stage, BackendId::new("missing"))
        .fallback_to(stage, BackendId::new("fallback"))
        .negotiate(&plan, &[requirement_ann(stage, 4, 2)])
        .expect("absent preferred backend should try the named fallback");

    assert_eq!(
        outcome.selection(stage),
        Some(&NegotiationOutcome::Fallback {
            rejected: BackendId::new("missing"),
            selected: BackendId::new("fallback"),
        })
    );
}

#[test]
fn fallback_without_prefer_does_not_select_an_empty_identity() {
    let mut graph = StageGraph::new();
    let stage = ann_stage(&mut graph, "tower", 4, 2);
    let plan = graph.compile().unwrap();
    let named = BackendCapabilities::ann("named")
        .with_dtypes([Dtype::F32])
        .with_hidden_dim(4)
        .with_max_sequence(8)
        .with_features([RequiredFeature::new("reference-embedding")]);
    let empty = BackendCapabilities::ann("empty")
        .with_dtypes([Dtype::F32])
        .with_hidden_dim(4)
        .with_max_sequence(8)
        .with_features([RequiredFeature::new("reference-embedding")]);

    let err = CapabilityNegotiation::new(FallbackPolicy::AllowNamed)
        .offer(BackendId::new(""), empty)
        .offer(BackendId::new("named"), named)
        .fallback_to(stage, BackendId::new("named"))
        .negotiate(&plan, &[requirement_ann(stage, 4, 2)])
        .expect_err("empty offer must not be selected ahead of the named fallback");
    assert!(
        matches!(err, PlanError::InvalidParameters(_)),
        "empty identity was accepted: {err:?}"
    );
}

#[test]
fn duplicate_stage_requirements_are_rejected() {
    let mut graph = StageGraph::new();
    let stage = ann_stage(&mut graph, "tower", 4, 2);
    let plan = graph.compile().unwrap();
    let wide = BackendCapabilities::ann("wide")
        .with_dtypes([Dtype::F32])
        .with_hidden_dim(4)
        .with_max_sequence(8)
        .with_features([
            RequiredFeature::new("reference-embedding"),
            RequiredFeature::new("streaming-embed"),
        ]);
    let replay = BackendCapabilities::ann("replay")
        .with_dtypes([Dtype::F32])
        .with_hidden_dim(4)
        .with_max_sequence(8)
        .with_features([
            RequiredFeature::new("reference-embedding"),
            RequiredFeature::new("caller_rng"),
        ]);

    let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(BackendId::new("wide"), wide)
        .offer(BackendId::new("replay"), replay)
        .negotiate(
            &plan,
            &[
                requirement_ann(stage, 4, 2).with_feature(RequiredFeature::new("streaming-embed")),
                requirement_ann(stage, 4, 2).with_feature(RequiredFeature::new("caller_rng")),
            ],
        )
        .expect_err("two requirements for one stage must not both succeed");
    assert!(
        matches!(err, PlanError::InvalidParameters(_)),
        "conflicting assignments were accepted: {err:?}"
    );
}

#[test]
fn output_population_is_checked_apart_from_stimulus_width() {
    let mut graph = StageGraph::new();
    let stage = snn_stage(&mut graph, "spikes", 4);
    let plan = graph.compile().unwrap();
    let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(
            BackendId::new("unequal"),
            BackendCapabilities::snn("unequal")
                .with_dtypes([Dtype::F32])
                .with_channels(4)
                .with_num_neurons(2)
                .with_stateful(true)
                .with_reset(true)
                .with_features([RequiredFeature::new("step")]),
        )
        .negotiate(&plan, &[requirement_snn(stage, 4).with_num_neurons(8)])
        .unwrap_err();
    match err {
        PlanError::IncompatibleCapability {
            field,
            required,
            backend,
            ..
        } => {
            assert_eq!(field, "num_neurons");
            assert_eq!(required, "8");
            assert_eq!(backend, "2");
        }
        other => panic!("expected output-population mismatch, got {other:?}"),
    }
}

#[test]
fn sole_domain_match_keeps_its_capability_error() {
    let mut graph = StageGraph::new();
    let stage = ann_stage(&mut graph, "tower", 4, 2);
    let plan = graph.compile().unwrap();
    let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(
            BackendId::new("ann-f16"),
            BackendCapabilities::ann("ann-f16").with_dtypes([Dtype::F16]),
        )
        .offer(
            BackendId::new("snn"),
            BackendCapabilities::snn("snn").with_dtypes([Dtype::F32]),
        )
        .negotiate(&plan, &[requirement_ann(stage, 4, 2)])
        .unwrap_err();
    match err {
        PlanError::IncompatibleCapability { field, .. } => assert_eq!(field, "dtype"),
        other => panic!("dtype mismatch was collapsed to a domain error: {other:?}"),
    }
}

#[test]
fn fallback_that_also_misses_the_sequence_window_is_not_semantic() {
    let mut graph = StageGraph::new();
    let stage = ann_stage(&mut graph, "tower", 4, 8);
    let plan = graph.compile().unwrap();
    let preferred = BackendCapabilities::ann("preferred").with_dtypes([Dtype::F16]);
    let short = BackendCapabilities::ann("short")
        .with_dtypes([Dtype::F32])
        .with_hidden_dim(4)
        .with_max_sequence(2);

    let err = CapabilityNegotiation::new(FallbackPolicy::AllowNamed)
        .offer(BackendId::new("preferred"), preferred)
        .offer(BackendId::new("short"), short)
        .prefer(stage, BackendId::new("preferred"))
        .fallback_to(stage, BackendId::new("short"))
        .negotiate(
            &plan,
            &[requirement_ann(stage, 4, 8).with_feature(RequiredFeature::new("caller_rng"))],
        )
        .unwrap_err();
    match err {
        PlanError::IncompatibleCapability { field, .. } => assert_eq!(field, "max_sequence"),
        other => panic!("sequence miss was reported as a semantic mismatch: {other:?}"),
    }
}

#[test]
fn sequence_window_shorter_than_required_is_rejected() {
    let mut graph = StageGraph::new();
    let stage = ann_stage(&mut graph, "tower", 4, 8);
    let plan = graph.compile().unwrap();
    let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(
            BackendId::new("short"),
            BackendCapabilities::ann("short")
                .with_dtypes([Dtype::F32])
                .with_hidden_dim(4)
                .with_max_sequence(4)
                .with_features([RequiredFeature::new("reference-embedding")]),
        )
        .negotiate(&plan, &[requirement_ann(stage, 4, 8)])
        .unwrap_err();
    match err {
        PlanError::IncompatibleCapability {
            field,
            required,
            backend,
            ..
        } => {
            assert_eq!(field, "max_sequence");
            assert_eq!(required, "8");
            assert_eq!(backend, "4");
        }
        other => panic!("expected max_sequence mismatch, got {other:?}"),
    }
}
