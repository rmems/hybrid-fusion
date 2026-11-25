// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stage-graph and backend-capability conformance matrix (RM-1813).
//!
//! Table-driven counterpart to `docs/conformance-matrix.md`: every
//! [`StageKind`] gets a row covering its domain mapping, the plan-level
//! outcome of pinning each [`ExecutionDomain`], and the negotiation outcome
//! for a matching, wrong-domain, and feature-missing backend.
//! [`expected_domains`] is an exhaustive match, so a new `StageKind` variant
//! fails to compile until its matrix row is added here and in the doc.

use std::collections::BTreeSet;

use hybrid_fusion::error::Result;
use hybrid_fusion::plan::{EdgeKind, ExecutionDomain, PlanError, StageId};
use hybrid_fusion::{
    BackendCapabilities, BackendId, CapabilityNegotiation, Dtype, ExpertRouteOutput, ExpertRouter,
    FallbackPolicy, HybridConfig, HybridError, HybridExecutionPlan, NegotiationOutcome,
    NeuroModulators, PortSpec, ProjectionMode, RateEncoder, RequiredFeature, SpikeCountDecoder,
    SpikingExpertRouter, SpikingNetwork, StageGraph, StageKind, StageRequirement,
};

/// Every stage kind, in declaration order.
const ALL_KINDS: [StageKind; 9] = [
    StageKind::Embedding,
    StageKind::Attention,
    StageKind::DenseMlp,
    StageKind::Transformer,
    StageKind::MoeRouter,
    StageKind::MoeExperts,
    StageKind::Adaptation,
    StageKind::SpikingBlock,
    StageKind::Readout,
];

/// Every domain a stage may be pinned to (`Auto` is only a request).
const ALL_DOMAINS: [ExecutionDomain; 4] = [
    ExecutionDomain::Ann,
    ExecutionDomain::Snn,
    ExecutionDomain::Moe,
    ExecutionDomain::Adapter,
];

/// The matrix's independent restatement of `StageKind::default_domain` /
/// `allowed_domains`. Deliberately exhaustive — adding an enum variant
/// without extending this table is a compile error.
fn expected_domains(kind: StageKind) -> (ExecutionDomain, &'static [ExecutionDomain]) {
    match kind {
        StageKind::Embedding => (ExecutionDomain::Ann, &[ExecutionDomain::Ann]),
        StageKind::Attention => (ExecutionDomain::Ann, &[ExecutionDomain::Ann]),
        StageKind::DenseMlp => (ExecutionDomain::Ann, &[ExecutionDomain::Ann]),
        StageKind::Transformer => (ExecutionDomain::Ann, &[ExecutionDomain::Ann]),
        StageKind::MoeRouter => (
            ExecutionDomain::Moe,
            &[ExecutionDomain::Moe, ExecutionDomain::Ann],
        ),
        StageKind::MoeExperts => (
            ExecutionDomain::Moe,
            &[ExecutionDomain::Moe, ExecutionDomain::Ann],
        ),
        StageKind::Adaptation => (
            ExecutionDomain::Adapter,
            &[ExecutionDomain::Adapter, ExecutionDomain::Ann],
        ),
        StageKind::SpikingBlock => (ExecutionDomain::Snn, &[ExecutionDomain::Snn]),
        StageKind::Readout => (
            ExecutionDomain::Ann,
            &[ExecutionDomain::Ann, ExecutionDomain::Adapter],
        ),
    }
}

/// A minimal offer in `domain`: F32-capable, no optional features.
fn caps_for(name: &str, domain: ExecutionDomain) -> BackendCapabilities {
    BackendCapabilities::ann(name)
        .with_domains([domain])
        .with_dtypes([Dtype::F32])
}

/// A compileable single stage of `kind` with a numerical F32 contract.
fn single_stage_plan(kind: StageKind) -> (HybridExecutionPlan, StageId) {
    let mut graph = StageGraph::new();
    let stage = graph.add_stage(
        "s",
        kind,
        PortSpec::f32_exact(&[4]),
        PortSpec::f32_exact(&[4]),
    );
    let plan = graph.compile().expect("single-stage graph compiles");
    (plan, stage)
}

// ---------------------------------------------------------------------------
// Stage kind → domain rows
// ---------------------------------------------------------------------------

#[test]
fn stage_domain_matrix_is_total() {
    // There is no stable enum iteration (`std::mem::variant_count` is
    // unstable, issue #73662), so completeness is enforced in two steps:
    // `expected_domains` fails to compile the moment a `StageKind` variant
    // appears (exhaustive match — add the row and this constant in the same
    // edit), and the count + uniqueness check below then rejects a stale or
    // duplicated `ALL_KINDS` row.
    const STAGE_KIND_COUNT: usize = 9; // number of `StageKind` variants
    assert_eq!(
        ALL_KINDS.len(),
        STAGE_KIND_COUNT,
        "a StageKind variant is missing its matrix row"
    );
    for (i, kind) in ALL_KINDS.iter().enumerate() {
        assert!(!ALL_KINDS[..i].contains(kind), "duplicate {kind:?} row");
    }

    for kind in ALL_KINDS {
        let (default, allowed) = expected_domains(kind);
        assert_eq!(kind.default_domain(), default, "{kind:?} default_domain");
        assert_eq!(kind.allowed_domains(), allowed, "{kind:?} allowed_domains");
        assert!(!allowed.contains(&ExecutionDomain::Auto));

        for domain in ALL_DOMAINS {
            let mut graph = StageGraph::new();
            let stage = graph.add_stage("s", kind, PortSpec::any(), PortSpec::any());
            if domain != default {
                assert!(graph.set_domain(stage, domain), "{kind:?} pin {domain:?}");
            }
            match graph.compile() {
                Ok(plan) => {
                    assert!(
                        allowed.contains(&domain),
                        "{kind:?} must reject an {domain:?} pin"
                    );
                    assert_eq!(
                        plan.resolved_domain(stage),
                        Some(domain),
                        "{kind:?} resolved domain"
                    );
                }
                Err(PlanError::InvalidDomain {
                    stage: got,
                    kind: got_kind,
                    domain: got_domain,
                    ..
                }) => {
                    assert!(!allowed.contains(&domain), "{kind:?} must allow {domain:?}");
                    assert_eq!(got, stage);
                    assert_eq!(got_kind, kind);
                    assert_eq!(got_domain, domain);
                }
                Err(other) => panic!("{kind:?} pinned {domain:?}: {other:?}"),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Negotiation outcome rows
// ---------------------------------------------------------------------------

#[test]
fn negotiation_outcome_matrix() {
    for kind in ALL_KINDS {
        let (domain, _) = expected_domains(kind);
        let (plan, stage) = single_stage_plan(kind);
        let requirement = || StageRequirement::new(stage, domain).with_dtype(Dtype::F32);

        // A backend advertising the resolved domain and dtype is selected.
        let report = CapabilityNegotiation::new(FallbackPolicy::Forbid)
            .offer(BackendId::new("fit"), caps_for("fit", domain))
            .negotiate(&plan, &[requirement()])
            .unwrap_or_else(|e| panic!("{kind:?} in {domain:?}: {e:?}"));
        assert_eq!(
            report.selection(stage),
            Some(&NegotiationOutcome::Selected(BackendId::new("fit"))),
            "{kind:?}"
        );

        // An offer in every other domain is a structured rejection naming
        // the required domain.
        for wrong in ALL_DOMAINS.into_iter().filter(|d| *d != domain) {
            let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
                .offer(BackendId::new("off-domain"), caps_for("off-domain", wrong))
                .negotiate(&plan, &[requirement()])
                .unwrap_err();
            assert!(
                matches!(
                    err,
                    PlanError::UnsupportedBackend {
                        stage: got,
                        domain: got_domain,
                        ..
                    } if got == stage && got_domain == domain
                ),
                "{kind:?} required {domain:?}, offered {wrong:?}: {err:?}"
            );
        }

        // A feature the domain-matching backend omits fails explicitly.
        let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
            .offer(BackendId::new("fit"), caps_for("fit", domain))
            .negotiate(
                &plan,
                &[requirement().with_feature(RequiredFeature::new("caller_rng"))],
            )
            .unwrap_err();
        assert!(
            matches!(err, PlanError::UnsupportedFeature { stage: got, .. } if got == stage),
            "{kind:?}: {err:?}"
        );
    }
}

#[test]
fn requirement_domain_must_equal_the_resolved_domain() {
    for kind in ALL_KINDS {
        let (default, allowed) = expected_domains(kind);

        // Auto-resolved stages reject a requirement naming any other domain,
        // even when the offer advertises it.
        let (plan, stage) = single_stage_plan(kind);
        for required in ALL_DOMAINS.into_iter().filter(|d| *d != default) {
            let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
                .offer(BackendId::new("b"), caps_for("b", required))
                .negotiate(&plan, &[StageRequirement::new(stage, required)])
                .unwrap_err();
            assert!(
                matches!(err, PlanError::InvalidParameters(_)),
                "{kind:?} required {required:?}: {err:?}"
            );
        }

        // A stage pinned to its second allowed domain likewise rejects the
        // default.
        if let Some(&alt) = allowed.iter().find(|d| **d != default) {
            let mut graph = StageGraph::new();
            let stage = graph.add_stage("s", kind, PortSpec::any(), PortSpec::any());
            assert!(graph.set_domain(stage, alt), "{kind:?} must pin to {alt:?}");
            let plan = graph.compile().unwrap();
            let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
                .offer(BackendId::new("b"), caps_for("b", default))
                .negotiate(&plan, &[StageRequirement::new(stage, default)])
                .unwrap_err();
            assert!(
                matches!(err, PlanError::InvalidParameters(_)),
                "{kind:?} pinned {alt:?} must reject a {default:?} requirement: {err:?}"
            );
        }
    }
}

#[test]
fn unpreferred_compatible_offers_are_rejected_as_ambiguous() {
    let (plan, stage) = single_stage_plan(StageKind::DenseMlp);
    let requirement = || StageRequirement::new(stage, ExecutionDomain::Ann).with_dtype(Dtype::F32);
    let negotiation = || {
        CapabilityNegotiation::new(FallbackPolicy::Forbid)
            .offer(BackendId::new("a"), caps_for("a", ExecutionDomain::Ann))
            .offer(BackendId::new("b"), caps_for("b", ExecutionDomain::Ann))
    };

    let err = negotiation()
        .negotiate(&plan, &[requirement()])
        .unwrap_err();
    match err {
        PlanError::UnsupportedBackend {
            stage: got, reason, ..
        } => {
            assert_eq!(got, stage);
            assert!(reason.contains("multiple compatible backends"), "{reason}");
        }
        other => panic!("expected UnsupportedBackend, got {other:?}"),
    }

    // prefer() disambiguates and is recorded.
    let report = negotiation()
        .prefer(stage, BackendId::new("b"))
        .negotiate(&plan, &[requirement()])
        .expect("prefer disambiguates");
    assert_eq!(
        report.selection(stage),
        Some(&NegotiationOutcome::Selected(BackendId::new("b")))
    );
}

#[test]
fn absent_preferred_backend_fails_structured_without_a_named_fallback() {
    let (plan, stage) = single_stage_plan(StageKind::DenseMlp);
    let err = CapabilityNegotiation::new(FallbackPolicy::AllowNamed)
        .offer(
            BackendId::new("other"),
            caps_for("other", ExecutionDomain::Ann),
        )
        .prefer(stage, BackendId::new("missing"))
        .negotiate(
            &plan,
            &[StageRequirement::new(stage, ExecutionDomain::Ann).with_dtype(Dtype::F32)],
        )
        .unwrap_err();
    match err {
        PlanError::UnsupportedBackend {
            stage: got, reason, ..
        } => {
            assert_eq!(got, stage);
            assert!(
                reason.contains("preferred backend 'missing' was not offered"),
                "{reason}"
            );
        }
        other => panic!("expected UnsupportedBackend, got {other:?}"),
    }
}

#[test]
fn requirement_for_a_missing_stage_is_unknown_stage() {
    let (plan, _) = single_stage_plan(StageKind::DenseMlp);
    // add_stage cannot mint a dangling id, but the serde shape can.
    let foreign: StageId = serde_json::from_str("7").unwrap();
    let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(BackendId::new("ann"), caps_for("ann", ExecutionDomain::Ann))
        .negotiate(
            &plan,
            &[StageRequirement::new(foreign, ExecutionDomain::Ann)],
        )
        .unwrap_err();
    assert!(matches!(
        err,
        PlanError::UnknownStage { stage } if stage == foreign
    ));
}

#[test]
fn edge_to_a_missing_stage_is_unknown_stage() {
    // Same trick: a deserialized graph may carry an edge endpoint no stage
    // owns; compile must reject it structurally, not panic.
    let json = r#"{
        "stages": [{
            "id": 0,
            "name": "a",
            "kind": "DenseMlp",
            "domain": "Auto",
            "input": {"dtype": null, "dims": []},
            "output": {"dtype": null, "dims": []},
            "attrs": {}
        }],
        "edges": [{"from": 0, "to": 1, "kind": "Forward"}],
        "allow_feedback": false
    }"#;
    let graph: StageGraph = serde_json::from_str(json).unwrap();
    assert!(matches!(
        graph.compile().unwrap_err(),
        PlanError::UnknownStage { stage } if stage.index() == 1
    ));
}

#[test]
fn selection_returns_none_for_unrequired_stages() {
    let mut graph = StageGraph::new();
    let required = graph.add_stage(
        "a",
        StageKind::DenseMlp,
        PortSpec::any(),
        PortSpec::f32_exact(&[4]),
    );
    let other = graph.add_stage(
        "b",
        StageKind::Readout,
        PortSpec::f32_exact(&[4]),
        PortSpec::any(),
    );
    graph.connect(required, other);
    let plan = graph.compile().unwrap();
    let report = CapabilityNegotiation::new(FallbackPolicy::Forbid)
        .offer(BackendId::new("ann"), caps_for("ann", ExecutionDomain::Ann))
        .negotiate(
            &plan,
            &[StageRequirement::new(required, ExecutionDomain::Ann)],
        )
        .unwrap();
    assert_eq!(report.selection(other), None);
    assert_eq!(report.outcomes().len(), 1);
    assert!(report.fallbacks().is_empty());
}

// ---------------------------------------------------------------------------
// Deterministic dry-run plan inspection
// ---------------------------------------------------------------------------

#[test]
fn forward_plan_describe_matches_the_golden_render() {
    let plan = HybridExecutionPlan::from_hybrid_config(&HybridConfig::tiny()).unwrap();
    let golden = [
        "HybridExecutionPlan {",
        "  [0] s0 'ann.transformer' kind=Transformer domain=Ann",
        "  [1] s1 'adapt.project_stimuli' kind=Adaptation domain=Adapter",
        "  [2] s2 'snn.step' kind=SpikingBlock domain=Snn",
        "  edge Forward s0 -> s1",
        "  edge Forward s1 -> s2",
        "}",
    ]
    .join("\n");
    assert_eq!(plan.describe(), golden);
}

#[test]
fn reverse_plan_describe_matches_the_golden_render() {
    let plan = HybridExecutionPlan::from_reverse_path(ProjectionMode::RateSum, 8, 16).unwrap();
    let golden = [
        "HybridExecutionPlan {",
        "  [0] s0 'snn.activity' kind=SpikingBlock domain=Snn",
        "  [1] s1 'adapt.project_activity' kind=Adaptation domain=Adapter",
        "  [2] s2 'moe.router' kind=MoeRouter domain=Moe",
        "  edge Forward s0 -> s1",
        "  edge Forward s1 -> s2",
        "}",
    ]
    .join("\n");
    assert_eq!(plan.describe(), golden);
}

#[test]
fn feedback_plan_describe_matches_the_golden_render() {
    let mut graph = StageGraph::new();
    let ann = graph.add_stage(
        "ann",
        StageKind::Transformer,
        PortSpec::any(),
        PortSpec::f32_exact(&[8]),
    );
    let snn = graph.add_stage(
        "snn",
        StageKind::SpikingBlock,
        PortSpec::f32_exact(&[8]),
        PortSpec::any(),
    );
    graph.allow_feedback(true);
    graph.connect(ann, snn);
    graph.connect_feedback(snn, ann);
    let plan = graph.compile().unwrap();
    let golden = [
        "HybridExecutionPlan {",
        "  [0] s0 'ann' kind=Transformer domain=Ann",
        "  [1] s1 'snn' kind=SpikingBlock domain=Snn",
        "  edge Forward s0 -> s1",
        "  edge Feedback s1 -> s0",
        "}",
    ]
    .join("\n");
    assert_eq!(plan.describe(), golden);
    assert_eq!(
        plan.edges()
            .iter()
            .filter(|e| e.kind == EdgeKind::Feedback)
            .count(),
        1
    );
}

#[test]
fn plan_rendering_is_stable_across_rebuild_and_serde_round_trip() {
    let plan = HybridExecutionPlan::from_hybrid_config(&HybridConfig::tiny()).unwrap();
    let rebuilt = HybridExecutionPlan::from_hybrid_config(&HybridConfig::tiny()).unwrap();
    assert_eq!(plan.describe(), rebuilt.describe());
    assert_eq!(plan.to_json().unwrap(), rebuilt.to_json().unwrap());
    let round_tripped = HybridExecutionPlan::from_json(&plan.to_json().unwrap()).unwrap();
    assert_eq!(round_tripped.describe(), plan.describe());
    assert_eq!(round_tripped.to_json().unwrap(), plan.to_json().unwrap());
}

// ---------------------------------------------------------------------------
// ANN↔SNN step contract (issue #61 / PR #65)
// ---------------------------------------------------------------------------

/// Minimal `SpikingNetwork`: implements only the mandatory `step`.
struct BareSnn;

impl SpikingNetwork for BareSnn {
    fn step(&mut self, _stimuli: &[f32], _modulators: &NeuroModulators) -> Result<Vec<usize>> {
        Ok(Vec::new())
    }

    fn num_channels(&self) -> usize {
        4
    }
}

#[test]
fn step_contract_defaults_fail_closed() {
    // The default report advertises exactly the mandatory surface: SNN
    // domain, F32, both width axes, a one-step window, and `step` alone.
    let caps = BareSnn.capabilities();
    assert_eq!(caps.domains, vec![ExecutionDomain::Snn]);
    assert_eq!(caps.dtypes, BTreeSet::from([Dtype::F32]));
    assert_eq!(caps.channels, Some(4));
    assert_eq!(caps.num_neurons, Some(4));
    assert_eq!(caps.max_sequence, Some(1));
    assert_eq!(
        caps.features.iter().map(|f| f.as_str()).collect::<Vec<_>>(),
        ["step"]
    );
    assert!(!caps.stateful && !caps.reset && !caps.streaming);
    assert!(caps.max_batch.is_none() && caps.hidden_dim.is_none());

    // Held-out ticks and reset fail closed instead of silently degrading to
    // the learning path or a no-op.
    assert!(matches!(
        BareSnn.step_frozen(&[0.0; 4], &NeuroModulators::default()),
        Err(HybridError::SnnStep(_))
    ));
    match BareSnn.reset() {
        Err(HybridError::SnnUnsupported(op)) => assert_eq!(op, "reset"),
        other => panic!("expected SnnUnsupported(\"reset\"), got {other:?}"),
    }
}

/// Router with a fixed two-logit gate, for the frozen-evaluation row.
struct GateRouter;

impl ExpertRouter for GateRouter {
    fn num_experts(&self) -> usize {
        2
    }

    fn top_k(&self) -> usize {
        1
    }

    fn route(&mut self, _embedding: &[f32]) -> Result<ExpertRouteOutput> {
        Ok(ExpertRouteOutput {
            expert_weights: vec![0.5; 2],
            selected_experts: vec![0],
            routing_entropy: None,
        })
    }

    fn gate_logits(&mut self, _embedding: &[f32]) -> Result<Vec<f32>> {
        Ok(vec![0.5; 2])
    }
}

/// SNN that records how many times `step` ran, so the test can prove the
/// frozen gate rejects before any backend work advances.
struct GateSnn {
    steps: usize,
}

impl SpikingNetwork for GateSnn {
    fn step(&mut self, _stimuli: &[f32], _modulators: &NeuroModulators) -> Result<Vec<usize>> {
        self.steps += 1;
        Ok(Vec::new())
    }

    fn num_channels(&self) -> usize {
        4
    }
}

#[test]
fn frozen_evaluation_is_a_capability_gate_not_a_fallback() {
    let mut router = SpikingExpertRouter::new(
        GateRouter,
        GateSnn { steps: 0 },
        RateEncoder::default(),
        SpikeCountDecoder,
    )
    .unwrap();
    let embedding = [0.5, 0.25];
    let err = router
        .forward_frozen(&embedding, &NeuroModulators::default())
        .unwrap_err();
    match err {
        HybridError::InvalidConfig(msg) => {
            assert!(msg.contains("frozen_evaluation"), "{msg}");
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
    // The rejection is a pre-step capability gate: the backend never ticked.
    assert_eq!(
        router.snn().steps,
        0,
        "frozen rejection must precede any backend work"
    );
    // The same backend's learning path is unaffected.
    router
        .forward(&embedding, &NeuroModulators::default())
        .unwrap();
    assert_eq!(router.snn().steps, 1);
}

// ---------------------------------------------------------------------------
// Reference backend reports (feature `backends`)
// ---------------------------------------------------------------------------

#[cfg(feature = "backends")]
mod backend_reports {
    use super::*;
    use hybrid_fusion::{SimpleSnn, SimpleTransformer, Transformer, TransformerConfig};

    #[test]
    fn simple_transformer_report_matches_the_trait_default() {
        let t = SimpleTransformer::new(TransformerConfig::tiny()).unwrap();
        let caps = t.capabilities();
        assert_eq!(caps.domains, vec![ExecutionDomain::Ann]);
        assert_eq!(caps.dtypes, BTreeSet::from([Dtype::F32]));
        assert_eq!(caps.hidden_dim, Some(t.dim()));
        assert_eq!(caps.max_sequence, Some(t.max_seq_len()));
        assert!(caps.features.is_empty());
        assert!(!caps.stateful && !caps.reset && !caps.streaming);
        assert!(caps.channels.is_none() && caps.num_neurons.is_none());
        assert!(caps.max_batch.is_none());
        // The advertised contract is what hidden_states honours.
        assert_eq!(t.hidden_states(&[0, 1, 2]).unwrap().shape(), &[3, t.dim()]);
    }

    #[test]
    fn mock_snn_report_lists_only_implemented_behaviors() {
        let mut snn = SimpleSnn::new(4);
        let caps = snn.capabilities();
        // Advertised capabilities are real: step ticks, reset is callable.
        assert!(caps.stateful && caps.reset);
        assert!(caps.features.iter().any(|f| f.as_str() == "step"));
        snn.step(&[0.1; 4], &NeuroModulators::default()).unwrap();
        <SimpleSnn as SpikingNetwork>::reset(&mut snn).unwrap();
        // Deliberately not advertised (module docs): the mock must not
        // satisfy a requirement that needs a feature it lacks.
        for feature in [
            "caller_rng",
            "plasticity",
            "neuromodulation",
            "frozen_evaluation",
        ] {
            assert!(
                !caps.features.iter().any(|f| f.as_str() == feature),
                "SimpleSnn must not advertise {feature}"
            );
        }
        assert!(matches!(
            snn.step_frozen(&[0.0; 4], &NeuroModulators::default()),
            Err(HybridError::SnnStep(_))
        ));
    }
}

// ---------------------------------------------------------------------------
// Mock-vs-real report fidelity (features `neuromod` + `backends`)
// ---------------------------------------------------------------------------

#[cfg(all(feature = "neuromod", feature = "backends"))]
mod real_backend_fidelity {
    use super::*;
    use hybrid_fusion::{NeuromodSnn, SimpleSnn};

    const SEED: u64 = 0xFEED_1813;

    fn feature_names(caps: &BackendCapabilities) -> BTreeSet<&str> {
        caps.features.iter().map(|f| f.as_str()).collect()
    }

    #[test]
    fn mock_and_real_snn_reports_share_the_contract_fields() {
        let mock = SimpleSnn::new(4).capabilities();
        let real = NeuromodSnn::with_seed(4, 4, SEED).unwrap().capabilities();

        // Every structural field the mock reports, the real backend reports
        // identically — the mock is a faithful stand-in for the step contract.
        assert_eq!(mock.domains, real.domains);
        assert_eq!(mock.dtypes, real.dtypes);
        assert_eq!(mock.channels, real.channels);
        assert_eq!(mock.num_neurons, real.num_neurons);
        assert_eq!(mock.max_sequence, real.max_sequence);
        assert_eq!(mock.stateful, real.stateful);
        assert_eq!(mock.reset, real.reset);
        assert_eq!(mock.streaming, real.streaming);
        assert_eq!(mock.max_batch, real.max_batch);
        assert_eq!(mock.device, real.device);

        // Identities differ (each report names its own backend), and the
        // real backend's extra features are exactly the documented set.
        assert_eq!(real.backend_name, "neuromod::SpikingNetwork");
        assert!(feature_names(&mock).is_subset(&feature_names(&real)));
        let extra: Vec<&str> = feature_names(&real)
            .difference(&feature_names(&mock))
            .copied()
            .collect();
        assert_eq!(
            extra,
            [
                "caller_rng",
                "frozen_evaluation",
                "neuromodulation",
                "plasticity"
            ],
            "SimpleSnn's deviations from NeuromodSnn are the documented set"
        );
    }

    #[test]
    fn real_snn_only_features_gate_negotiation_not_the_mock() {
        // A SpikingBlock requirement needing each real-only feature selects
        // the real backend and rejects the mock: the reports are not
        // interchangeable.
        let (plan, stage) = single_stage_plan(StageKind::SpikingBlock);
        let mock = SimpleSnn::new(4).capabilities();
        let real = NeuromodSnn::with_seed(4, 4, SEED).unwrap().capabilities();

        for feature in [
            "caller_rng",
            "plasticity",
            "neuromodulation",
            "frozen_evaluation",
        ] {
            let requirement = StageRequirement::new(stage, ExecutionDomain::Snn)
                .with_dtype(Dtype::F32)
                .with_feature(RequiredFeature::new(feature));
            let report = CapabilityNegotiation::new(FallbackPolicy::Forbid)
                .offer(BackendId::new("neuromod"), real.clone())
                .negotiate(&plan, std::slice::from_ref(&requirement))
                .unwrap_or_else(|e| panic!("NeuromodSnn must satisfy {feature}: {e:?}"));
            assert_eq!(
                report.selection(stage),
                Some(&NegotiationOutcome::Selected(BackendId::new("neuromod")))
            );
            let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
                .offer(BackendId::new("mock"), mock.clone())
                .negotiate(&plan, &[requirement])
                .unwrap_err();
            assert!(
                matches!(err, PlanError::UnsupportedFeature { stage: got, .. } if got == stage),
                "SimpleSnn must not satisfy {feature}: {err:?}"
            );
        }
    }

    #[test]
    fn advertised_real_features_are_implemented() {
        // frozen_evaluation: the advertised path exists (not the fail-closed
        // default); reset: callable as advertised; caller_rng: fixed seeds
        // replay identically.
        let mut snn = NeuromodSnn::with_seed(4, 4, SEED).unwrap();
        snn.step_frozen(&[0.5; 4], &NeuroModulators::default())
            .unwrap();
        <NeuromodSnn as SpikingNetwork>::reset(&mut snn).unwrap();
        let run = || {
            NeuromodSnn::with_seed(4, 4, SEED)
                .unwrap()
                .step(&[0.9, 0.8, 0.7, 0.6], &NeuroModulators::default())
                .unwrap()
        };
        assert_eq!(run(), run());
        // plasticity + neuromodulation behavioural anchors live in
        // src/neuromod_adapter.rs tests; the negotiation row above is the
        // matrix's handle on them.
    }
}
