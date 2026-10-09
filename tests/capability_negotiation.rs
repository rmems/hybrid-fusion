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
// - requirements contradict the compiled stage's fixed port contract;
// - reset's named-feature spelling disagrees with its typed capability;
// - a default SNN report omits the mandatory one-step temporal window.

//! Capability negotiation for hybrid stages (issue #41 / RM-1805).
//!
//! Mock backends exercise substitution, policy-driven fallback, and semantic
//! incompatibility. Discovery is asserted side-effect free by comparing
//! backend counters before and after `capabilities()`.

use hybrid_fusion::error::Result;
use hybrid_fusion::plan::{DimSpec, ExecutionDomain, PlanError, StageId};
use hybrid_fusion::{
    BackendCapabilities, BackendId, CapabilityNegotiation, Dtype, FallbackPolicy, HybridConfig,
    HybridExecutionPlan, NegotiationOutcome, NeuroModulators, PortSpec, ProjectionMode,
    RequiredFeature, SpikingNetwork, StageGraph, StageKind, StageRequirement, Tensor, Transformer,
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
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
    fn capabilities(&self) -> BackendCapabilities {
        self.caps.clone()
    }
}

struct DefaultCapsSnn;

impl SpikingNetwork for DefaultCapsSnn {
    fn step(&mut self, _stimuli: &[f32], _modulators: &NeuroModulators) -> Result<Vec<usize>> {
        Ok(Vec::new())
    }

    fn num_channels(&self) -> usize {
        4
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
fn requirement_must_match_fixed_snn_input_port() {
    let mut graph = StageGraph::new();
    let stage = snn_stage(&mut graph, "spikes", 4);
    let plan = graph.compile().expect("single stage");
    let caps = BackendCapabilities::snn("eight-channel")
        .with_dtypes([Dtype::F32])
        .with_channels(8);

    let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(BackendId::new("eight-channel"), caps)
        .negotiate(
            &plan,
            &[StageRequirement::new(stage, ExecutionDomain::Snn)
                .with_dtype(Dtype::F32)
                .with_channels(8)],
        )
        .expect_err("requirement must not contradict the stage input port");

    match err {
        PlanError::InvalidParameters(reason) => {
            assert!(reason.contains("channels") && reason.contains("4") && reason.contains("8"));
        }
        other => panic!("expected InvalidParameters, got {other:?}"),
    }
}

#[test]
fn requirement_must_match_fixed_transformer_output_port() {
    let mut graph = StageGraph::new();
    let stage = ann_stage(&mut graph, "tower", 4, 2);
    let plan = graph.compile().expect("single stage");
    let caps = BackendCapabilities::ann("wrong-contract")
        .with_dtypes([Dtype::F16])
        .with_hidden_dim(8)
        .with_max_sequence(3);

    for requirement in [
        StageRequirement::new(stage, ExecutionDomain::Ann).with_dtype(Dtype::F16),
        StageRequirement::new(stage, ExecutionDomain::Ann).with_hidden_dim(8),
        StageRequirement::new(stage, ExecutionDomain::Ann).with_max_sequence(3),
    ] {
        let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
            .offer(BackendId::new("wrong-contract"), caps.clone())
            .negotiate(&plan, &[requirement])
            .expect_err("requirement must not contradict the stage output port");
        assert!(matches!(err, PlanError::InvalidParameters(_)), "{err:?}");
    }
}

#[test]
fn every_stage_kind_validates_its_numerical_port_dtype() {
    let cases = [
        (StageKind::Embedding, ExecutionDomain::Ann),
        (StageKind::Attention, ExecutionDomain::Ann),
        (StageKind::DenseMlp, ExecutionDomain::Ann),
        (StageKind::MoeRouter, ExecutionDomain::Moe),
        (StageKind::MoeExperts, ExecutionDomain::Moe),
        (StageKind::Adaptation, ExecutionDomain::Adapter),
        (StageKind::Readout, ExecutionDomain::Ann),
    ];

    for (kind, domain) in cases {
        let mut graph = StageGraph::new();
        let stage = graph.add_stage(
            format!("{kind:?}"),
            kind,
            PortSpec::f32_exact(&[4]),
            PortSpec::f32_exact(&[4]),
        );
        let plan = graph.compile().expect("single numerical stage");
        let caps = BackendCapabilities::ann("f16-only")
            .with_domains([domain])
            .with_dtypes([Dtype::F16]);

        let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
            .offer(BackendId::new("f16-only"), caps)
            .negotiate(
                &plan,
                &[StageRequirement::new(stage, domain).with_dtype(Dtype::F16)],
            )
            .expect_err("requirement must match the compiled numerical port dtype");
        assert!(
            matches!(err, PlanError::InvalidParameters(_)),
            "{kind:?} bypassed its port contract: {err:?}"
        );
    }
}

#[test]
fn embedding_requirement_uses_activation_not_token_id_dtype() {
    let mut graph = StageGraph::new();
    let stage = graph.add_stage(
        "embedding",
        StageKind::Embedding,
        PortSpec {
            dtype: Some(Dtype::U32),
            dims: vec![DimSpec::Symbolic("seq".into())],
        },
        PortSpec::f32_exact(&[4]),
    );
    let plan = graph.compile().expect("embedding stage");

    CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(
            BackendId::new("embedding"),
            BackendCapabilities::ann("embedding")
                .with_dtypes([Dtype::F32])
                .with_hidden_dim(4),
        )
        .negotiate(
            &plan,
            &[StageRequirement::new(stage, ExecutionDomain::Ann)
                .with_dtype(Dtype::F32)
                .with_hidden_dim(4)],
        )
        .expect("token IDs are not the embedding backend's activation dtype");
}

#[test]
fn numerical_stage_width_does_not_match_an_unrelated_axis() {
    let mut graph = StageGraph::new();
    let stage = graph.add_stage(
        "mlp",
        StageKind::DenseMlp,
        PortSpec::f32_exact(&[2, 4]),
        PortSpec::f32_exact(&[2, 4]),
    );
    let plan = graph.compile().expect("dense stage");

    let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(
            BackendId::new("wrong-width"),
            BackendCapabilities::ann("wrong-width")
                .with_dtypes([Dtype::F32])
                .with_hidden_dim(2),
        )
        .negotiate(
            &plan,
            &[StageRequirement::new(stage, ExecutionDomain::Ann)
                .with_dtype(Dtype::F32)
                .with_hidden_dim(2)],
        )
        .expect_err("sequence axis must not satisfy hidden width");
    assert!(matches!(err, PlanError::InvalidParameters(_)), "{err:?}");
}

#[test]
fn resolved_symbolic_width_constrains_negotiation() {
    let mut graph = StageGraph::new();
    let producer = graph.add_stage(
        "producer",
        StageKind::Adaptation,
        PortSpec::any(),
        PortSpec::f32_exact(&[4]),
    );
    let snn = graph.add_stage(
        "spikes",
        StageKind::SpikingBlock,
        PortSpec::f32(vec![DimSpec::Symbolic("width".into())]),
        PortSpec::any(),
    );
    graph.connect(producer, snn);
    let plan = graph.compile().expect("symbol resolves to producer width");

    let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(
            BackendId::new("eight-channel"),
            BackendCapabilities::snn("eight-channel")
                .with_dtypes([Dtype::F32])
                .with_channels(8),
        )
        .negotiate(
            &plan,
            &[StageRequirement::new(snn, ExecutionDomain::Snn)
                .with_dtype(Dtype::F32)
                .with_channels(8)],
        )
        .expect_err("resolved width four must reject an eight-channel backend");
    assert!(matches!(err, PlanError::InvalidParameters(_)), "{err:?}");
}

#[test]
fn reverse_path_neuron_metadata_constrains_negotiation() {
    let plan = HybridExecutionPlan::from_reverse_path(ProjectionMode::RateSum, 8, 4)
        .expect("reverse plan");
    let activity = plan.stage_by_name("snn.activity").unwrap().id;

    let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(
            BackendId::new("four-neuron"),
            BackendCapabilities::snn("four-neuron").with_num_neurons(4),
        )
        .negotiate(
            &plan,
            &[StageRequirement::new(activity, ExecutionDomain::Snn).with_num_neurons(4)],
        )
        .expect_err("reverse activity population is fixed at eight");
    assert!(matches!(err, PlanError::InvalidParameters(_)), "{err:?}");
}

#[test]
fn canonical_plan_checks_backend_dtype_on_numerical_ports() {
    let cfg = HybridConfig::tiny();
    let plan = hybrid_fusion::HybridExecutionPlan::from_hybrid_config(&cfg).unwrap();
    let ann = plan.stage_by_name("ann.transformer").unwrap().id;
    let snn = plan.stage_by_name("snn.step").unwrap().id;

    CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(
            BackendId::new("ann"),
            BackendCapabilities::ann("ann").with_dtypes([Dtype::F32]),
        )
        .offer(
            BackendId::new("snn"),
            BackendCapabilities::snn("snn").with_dtypes([Dtype::F32]),
        )
        .negotiate(
            &plan,
            &[
                StageRequirement::new(ann, ExecutionDomain::Ann).with_dtype(Dtype::F32),
                StageRequirement::new(snn, ExecutionDomain::Snn).with_dtype(Dtype::F32),
            ],
        )
        .expect("U32 token IDs must not conflict with the F32 Transformer output dtype");
}

#[test]
fn canonical_plan_metadata_constrains_backend_dimensions() {
    let cfg = HybridConfig::tiny();
    let plan = hybrid_fusion::HybridExecutionPlan::from_hybrid_config(&cfg).unwrap();
    let ann = plan.stage_by_name("ann.transformer").unwrap().id;
    let snn = plan.stage_by_name("snn.step").unwrap().id;

    for requirement in [
        StageRequirement::new(ann, ExecutionDomain::Ann).with_hidden_dim(cfg.transformer.dim + 1),
        StageRequirement::new(snn, ExecutionDomain::Snn).with_num_neurons(cfg.snn_lif_neurons + 1),
    ] {
        let caps = match requirement.domain() {
            ExecutionDomain::Ann => {
                BackendCapabilities::ann("wrong-size").with_hidden_dim(cfg.transformer.dim + 1)
            }
            ExecutionDomain::Snn => {
                BackendCapabilities::snn("wrong-size").with_num_neurons(cfg.snn_lif_neurons + 1)
            }
            domain => panic!("unexpected test domain {domain:?}"),
        };
        let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
            .offer(BackendId::new("wrong-size"), caps)
            .negotiate(&plan, &[requirement])
            .expect_err("canonical stage metadata must constrain backend dimensions");
        assert!(matches!(err, PlanError::InvalidParameters(_)), "{err:?}");
    }
}

#[test]
fn canonical_transformer_requires_its_configured_sequence_capacity() {
    let mut cfg = HybridConfig::tiny();
    cfg.transformer.max_seq_len = 8;
    let plan = HybridExecutionPlan::from_hybrid_config(&cfg).unwrap();
    let transformer = plan.stage_by_name("ann.transformer").unwrap().id;

    let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(
            BackendId::new("short"),
            BackendCapabilities::ann("short")
                .with_dtypes([Dtype::F32])
                .with_max_sequence(4),
        )
        .negotiate(
            &plan,
            &[StageRequirement::new(transformer, ExecutionDomain::Ann)
                .with_dtype(Dtype::F32)
                .with_max_sequence(4)],
        )
        .expect_err("backend capacity must cover the canonical plan");
    assert!(matches!(err, PlanError::InvalidParameters(_)), "{err:?}");
}

#[test]
fn canonical_forward_projector_requires_transformer_hidden_width() {
    let cfg = HybridConfig::tiny();
    let plan = HybridExecutionPlan::from_hybrid_config(&cfg).unwrap();
    let projector = plan.stage_by_name("adapt.project_stimuli").unwrap().id;
    let wrong = cfg.transformer.dim + 1;

    let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(
            BackendId::new("wrong-width"),
            BackendCapabilities::adapter("wrong-width")
                .with_dtypes([Dtype::F32])
                .with_hidden_dim(wrong),
        )
        .negotiate(
            &plan,
            &[StageRequirement::new(projector, ExecutionDomain::Adapter)
                .with_dtype(Dtype::F32)
                .with_hidden_dim(wrong)],
        )
        .expect_err("forward projector width must match its transformer source");
    assert!(matches!(err, PlanError::InvalidParameters(_)), "{err:?}");
}

#[test]
fn canonical_reverse_projector_requires_activity_population() {
    let plan = HybridExecutionPlan::from_reverse_path(ProjectionMode::RateSum, 8, 4)
        .expect("reverse plan");
    let projector = plan.stage_by_name("adapt.project_activity").unwrap().id;

    let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(
            BackendId::new("four-neuron"),
            BackendCapabilities::adapter("four-neuron").with_num_neurons(4),
        )
        .negotiate(
            &plan,
            &[StageRequirement::new(projector, ExecutionDomain::Adapter).with_num_neurons(4)],
        )
        .expect_err("reverse projector population must match its activity source");
    assert!(matches!(err, PlanError::InvalidParameters(_)), "{err:?}");
}

#[test]
fn reset_named_feature_uses_the_typed_reset_capability() {
    let mut graph = StageGraph::new();
    let stage = snn_stage(&mut graph, "spikes", 4);
    let plan = graph.compile().expect("single stage");

    let report = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(
            BackendId::new("resettable"),
            BackendCapabilities::snn("resettable")
                .with_dtypes([Dtype::F32])
                .with_channels(4)
                .with_reset(true),
        )
        .negotiate(
            &plan,
            &[StageRequirement::new(stage, ExecutionDomain::Snn)
                .with_feature(RequiredFeature::new("reset"))],
        )
        .expect("reset feature alias should use the typed reset capability");

    assert_eq!(
        report.selection(stage),
        Some(&NegotiationOutcome::Selected(BackendId::new("resettable")))
    );

    let normalized_caps = BackendCapabilities::snn("legacy-feature")
        .with_dtypes([Dtype::F32])
        .with_channels(4)
        .with_features([RequiredFeature::new("reset")]);
    assert!(normalized_caps.reset);
    assert!(normalized_caps.features.is_empty());
    let feature_report = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(BackendId::new("legacy-feature"), normalized_caps)
        .negotiate(
            &plan,
            &[StageRequirement::new(stage, ExecutionDomain::Snn).requires_reset()],
        )
        .expect("reset feature report should normalize to the typed capability");
    assert_eq!(
        feature_report.selection(stage),
        Some(&NegotiationOutcome::Selected(BackendId::new(
            "legacy-feature"
        )))
    );
}

#[test]
fn deserialized_reset_feature_normalizes_to_typed_capability() {
    let mut value = serde_json::to_value(BackendCapabilities::snn("serialized")).unwrap();
    value["reset"] = serde_json::json!(false);
    value["features"] = serde_json::json!(["reset"]);

    let caps: BackendCapabilities = serde_json::from_value(value).unwrap();

    assert!(caps.reset);
    assert!(caps.features.is_empty());
}

#[test]
fn dtype_conversion_requires_backend_support_for_both_ports() {
    let mut graph = StageGraph::new();
    let stage = graph.add_stage(
        "cast",
        StageKind::Adaptation,
        PortSpec {
            dtype: Some(Dtype::F16),
            dims: vec![DimSpec::Fixed(4)],
        },
        PortSpec::f32_exact(&[4]),
    );
    let plan = graph.compile().expect("dtype conversion stage");
    let requirement = StageRequirement::new(stage, ExecutionDomain::Adapter)
        .with_dtype(Dtype::F16)
        .with_channels(4);

    CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(
            BackendId::new("converter"),
            BackendCapabilities::adapter("converter")
                .with_dtypes([Dtype::F16, Dtype::F32])
                .with_channels(4),
        )
        .negotiate(&plan, std::slice::from_ref(&requirement))
        .expect("backend supports both conversion port dtypes");

    let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(
            BackendId::new("input-only"),
            BackendCapabilities::adapter("input-only")
                .with_dtypes([Dtype::F16])
                .with_channels(4),
        )
        .negotiate(&plan, &[requirement])
        .expect_err("backend missing the output dtype must be rejected");
    assert!(matches!(err, PlanError::IncompatibleCapability { .. }));
}

#[test]
fn default_snn_report_advertises_one_step_window() {
    let caps = DefaultCapsSnn.capabilities();
    assert_eq!(caps.max_sequence, Some(1));
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
fn fallback_without_prefer_is_invalid_before_candidate_evaluation() {
    let mut graph = StageGraph::new();
    let stage = ann_stage(&mut graph, "tower", 4, 2);
    let plan = graph.compile().unwrap();
    let fallback = BackendCapabilities::ann("fallback")
        .with_dtypes([Dtype::F32])
        .with_hidden_dim(4)
        .with_max_sequence(8)
        .with_features([RequiredFeature::new("reference-embedding")]);

    let err = CapabilityNegotiation::new(FallbackPolicy::AllowNamed)
        .offer(BackendId::new("fallback"), fallback)
        .fallback_to(stage, BackendId::new("fallback"))
        .negotiate(&plan, &[requirement_ann(stage, 4, 2)])
        .expect_err("fallback_to without prefer must be invalid configuration");
    assert!(
        matches!(err, PlanError::InvalidParameters(_)),
        "fallback-only configuration returned the wrong error: {err:?}"
    );
}

#[test]
fn fallback_without_prefer_is_rejected_even_for_an_unrequired_stage() {
    let mut graph = StageGraph::new();
    let required = ann_stage(&mut graph, "required", 4, 2);
    let unused = graph.add_stage(
        "unused-policy-target",
        StageKind::DenseMlp,
        PortSpec::f32_exact(&[2, 4]),
        PortSpec::f32_exact(&[4]),
    );
    graph.connect(required, unused);
    let plan = graph.compile().expect("connected plan");

    let err = CapabilityNegotiation::new(FallbackPolicy::AllowNamed)
        .offer(
            BackendId::new("ann"),
            BackendCapabilities::ann("ann")
                .with_dtypes([Dtype::F32])
                .with_hidden_dim(4)
                .with_max_sequence(2)
                .with_features([RequiredFeature::new("reference-embedding")]),
        )
        .fallback_to(unused, BackendId::new("ann"))
        .negotiate(&plan, &[requirement_ann(required, 4, 2)])
        .expect_err("all malformed preference entries must be rejected");
    assert!(matches!(err, PlanError::InvalidParameters(_)), "{err:?}");
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
