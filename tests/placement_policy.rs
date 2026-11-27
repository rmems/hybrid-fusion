// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dry-run placement planning (issue #24). File-free by contract: policies
//! see stage names, kinds, port contracts, and capability reports — no
//! weights, manifests, paths, or loaders. Imports come from the crate root so
//! the public re-exports stay honest.

use hybrid_fusion::{
    AnnTopology, BackendCapabilities, EdgeKind, ExecutionDomain, HybridExecutionPlan,
    HybridTopology, Placement, PlacementDecision, PlacementPlanner, PlacementPolicy,
    PlacementReport, PlanError, PortSpec, SnnTopology, StageGraph, StageId, StageKind,
};

// ---------------------------------------------------------------------------
// Mock policies
// ---------------------------------------------------------------------------

/// Keeps every discovered stage on its existing domain assignment.
struct PreserveAll;

impl PlacementPolicy for PreserveAll {
    fn name(&self) -> &str {
        "preserve-all"
    }

    fn place(
        &self,
        graph: &StageGraph,
        _backends: &[BackendCapabilities],
    ) -> Result<Vec<PlacementDecision>, PlanError> {
        Ok(graph
            .stages()
            .iter()
            .map(|s| PlacementDecision::new(s.id, Placement::Preserve))
            .collect())
    }
}

/// Routes stages to spiking hardware by kind; fails closed on a kind it does
/// not understand rather than guessing a domain.
struct SnnOffload {
    snn_kinds: &'static [StageKind],
}

impl PlacementPolicy for SnnOffload {
    fn name(&self) -> &str {
        "snn-offload"
    }

    fn place(
        &self,
        graph: &StageGraph,
        _backends: &[BackendCapabilities],
    ) -> Result<Vec<PlacementDecision>, PlanError> {
        graph
            .stages()
            .iter()
            .map(|s| {
                if self.snn_kinds.contains(&s.kind) {
                    Ok(PlacementDecision::new(s.id, Placement::Snn))
                } else {
                    Ok(PlacementDecision::new(s.id, Placement::Preserve))
                }
            })
            .collect()
    }
}

/// A model-family policy vocabulary adapted into `Placement` — the RM-1817
/// pattern. `Tier` stands in for a foreign precision enum (e.g. grok-ozempic's
/// `TensorPrecision`): it lives outside this crate, maps onto `Placement`, and
/// survives in the report as namespaced metadata. The core enums never name it.
enum Tier {
    Preserve,
    HalfDense,
    TernarySpike,
}

struct TierAdapter;

impl PlacementPolicy for TierAdapter {
    fn name(&self) -> &str {
        "tier-adapter"
    }

    fn place(
        &self,
        graph: &StageGraph,
        _backends: &[BackendCapabilities],
    ) -> Result<Vec<PlacementDecision>, PlanError> {
        graph
            .stages()
            .iter()
            .map(|s| {
                let tier = match s.kind {
                    StageKind::SpikingBlock => Tier::TernarySpike,
                    StageKind::Adaptation => Tier::HalfDense,
                    _ => Tier::Preserve,
                };
                let (placement, label) = match tier {
                    Tier::Preserve => (Placement::Preserve, "preserve"),
                    Tier::HalfDense => (Placement::Hybrid, "half_dense"),
                    Tier::TernarySpike => (Placement::Snn, "ternary_spike"),
                };
                Ok(PlacementDecision::new(s.id, placement).with_attr("family.tier", label))
            })
            .collect()
    }
}

/// Decides a placement no backend supports.
struct Impossible;

impl PlacementPolicy for Impossible {
    fn name(&self) -> &str {
        "impossible"
    }

    fn place(
        &self,
        graph: &StageGraph,
        _backends: &[BackendCapabilities],
    ) -> Result<Vec<PlacementDecision>, PlanError> {
        Ok(graph
            .stages()
            .iter()
            .map(|s| PlacementDecision::new(s.id, Placement::Snn))
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// A generic discovered model: embedding -> dense block -> boundary
/// adaptation -> spiking block -> readout. Shapes are symbolic where a real
/// discovery would pin checkpoint widths; nothing here is a Grok type.
fn discovered_graph() -> StageGraph {
    let mut g = StageGraph::new();
    let embed = g.add_stage(
        "blk.embed",
        StageKind::Embedding,
        PortSpec::any(),
        PortSpec::f32(vec![hybrid_fusion::DimSpec::Symbolic("hidden".into())]),
    );
    let dense = g.add_stage(
        "blk.mlp",
        StageKind::DenseMlp,
        PortSpec::f32(vec![hybrid_fusion::DimSpec::Symbolic("hidden".into())]),
        PortSpec::f32(vec![hybrid_fusion::DimSpec::Symbolic("hidden".into())]),
    );
    let adapt = g.add_stage(
        "blk.adapt",
        StageKind::Adaptation,
        PortSpec::f32(vec![hybrid_fusion::DimSpec::Symbolic("hidden".into())]),
        PortSpec::f32(vec![hybrid_fusion::DimSpec::Symbolic("channels".into())]),
    );
    let snn = g.add_stage(
        "blk.snn",
        StageKind::SpikingBlock,
        PortSpec::f32(vec![hybrid_fusion::DimSpec::Symbolic("channels".into())]),
        PortSpec::any(),
    );
    let read = g.add_stage(
        "blk.readout",
        StageKind::Readout,
        PortSpec::any(),
        PortSpec::any(),
    );
    g.connect(embed, dense);
    g.connect(dense, adapt);
    g.connect(adapt, snn);
    g.connect(snn, read);
    g
}

fn topology() -> HybridTopology {
    HybridTopology::new(
        AnnTopology::new(64, 32, Some("mock-family".into())).unwrap(),
        SnnTopology::new(16, 8).unwrap(),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Contract tests
// ---------------------------------------------------------------------------

#[test]
fn preserve_policy_reproduces_the_canonical_plan() {
    // dry_run_plan over the canonical topology plan: Preserve keeps every
    // kind's default domain, so the placed plan equals the compiled one.
    let plan = HybridExecutionPlan::from_topology(&topology()).unwrap();
    let report = PlacementPlanner::new(PreserveAll)
        .dry_run_plan(&plan)
        .unwrap();
    assert_eq!(report.plan(), &plan);
    assert_eq!(report.policy(), "preserve-all");
    for d in report.decisions() {
        assert_eq!(d.placement, Placement::Preserve);
    }
}

#[test]
fn preserve_policy_keeps_existing_pins_and_feedback() {
    let mut g = discovered_graph();
    g.allow_feedback(true);
    let snn = g.stage_by_name("blk.snn").expect("snn").id;
    let adapt = g.stage_by_name("blk.adapt").expect("adapt").id;
    assert!(g.set_domain(snn, ExecutionDomain::Snn));
    g.connect_feedback(snn, adapt);
    let plan = g.compile().expect("compiled graph with pin and feedback");
    assert_eq!(plan.resolved_domain(snn), Some(ExecutionDomain::Snn));
    assert!(
        plan.edges()
            .iter()
            .any(|e| e.kind == EdgeKind::Feedback && e.from == snn && e.to == adapt)
    );

    let report = PlacementPlanner::new(PreserveAll)
        .dry_run_plan(&plan)
        .expect("preserve dry_run_plan");
    assert_eq!(report.plan(), &plan);
    assert_eq!(
        report.plan().resolved_domain(snn),
        Some(ExecutionDomain::Snn)
    );
    assert!(
        report
            .plan()
            .edges()
            .iter()
            .any(|e| e.kind == EdgeKind::Feedback && e.from == snn && e.to == adapt)
    );
}

#[test]
fn policy_metadata_rides_decision_attrs_not_core_enums() {
    let graph = discovered_graph();
    let report: PlacementReport = PlacementPlanner::new(TierAdapter).dry_run(&graph).unwrap();
    let plan = report.plan();

    let adapt = plan.stage_by_name("blk.adapt").unwrap();
    let snn = plan.stage_by_name("blk.snn").unwrap();
    let embed = plan.stage_by_name("blk.embed").unwrap();

    // The family vocabulary is inspectable as metadata on the compiled plan.
    assert_eq!(
        adapt.attrs.get("family.tier").map(String::as_str),
        Some("half_dense")
    );
    assert_eq!(
        snn.attrs.get("family.tier").map(String::as_str),
        Some("ternary_spike")
    );
    // …while the resolved domains stay in the generic vocabulary.
    assert_eq!(
        plan.resolved_domain(adapt.id),
        Some(ExecutionDomain::Adapter)
    );
    assert_eq!(plan.resolved_domain(snn.id), Some(ExecutionDomain::Snn));
    assert_eq!(plan.resolved_domain(embed.id), Some(ExecutionDomain::Ann));

    let decision = report.decision_for(adapt.id).unwrap();
    assert_eq!(decision.placement, Placement::Hybrid);
}

#[test]
fn snn_placement_requires_an_snn_capable_stage_kind() {
    // A uniform Snn placement pins DenseMlp/Embedding/Adaptation outside
    // their allowed domain set — an explicit failure, never a silent re-route.
    let err = PlacementPlanner::new(Impossible)
        .dry_run(&discovered_graph())
        .unwrap_err();
    match err {
        PlanError::InvalidDomain { domain, .. } => assert_eq!(domain, ExecutionDomain::Snn),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn legal_snn_offload_plans_the_full_graph() {
    // SpikingBlock is the only Snn-capable kind; everything else is preserved.
    let policy = SnnOffload {
        snn_kinds: &[StageKind::SpikingBlock],
    };
    let report = PlacementPlanner::new(policy)
        .offer(BackendCapabilities::ann("dense-host"))
        .offer(BackendCapabilities::adapter("boundary"))
        .offer(BackendCapabilities::snn("spike-rt"))
        .dry_run(&discovered_graph())
        .unwrap();
    let snn = report.plan().stage_by_name("blk.snn").unwrap();
    assert_eq!(
        report.plan().resolved_domain(snn.id),
        Some(ExecutionDomain::Snn)
    );
}

#[test]
fn placement_fails_when_no_backend_advertises_the_domain() {
    // SpikingBlock legally resolves to Snn, but only ANN/Adapter backends are
    // offered — the dry run fails with UnsupportedBackend, not a fake plan.
    let policy = SnnOffload {
        snn_kinds: &[StageKind::SpikingBlock],
    };
    let err = PlacementPlanner::new(policy)
        .offer(BackendCapabilities::ann("dense-host"))
        .offer(BackendCapabilities::adapter("boundary"))
        .dry_run(&discovered_graph())
        .unwrap_err();
    match err {
        PlanError::UnsupportedBackend { domain, .. } => {
            assert_eq!(domain, ExecutionDomain::Snn)
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn missing_and_duplicate_decisions_fail_explicitly() {
    struct SkipOne;
    impl PlacementPolicy for SkipOne {
        fn name(&self) -> &str {
            "skip-one"
        }
        fn place(
            &self,
            graph: &StageGraph,
            _b: &[BackendCapabilities],
        ) -> Result<Vec<PlacementDecision>, PlanError> {
            Ok(graph
                .stages()
                .iter()
                .skip(1)
                .map(|s| PlacementDecision::new(s.id, Placement::Preserve))
                .collect())
        }
    }
    assert!(matches!(
        PlacementPlanner::new(SkipOne).dry_run(&discovered_graph()),
        Err(PlanError::InvalidPlacement { .. })
    ));

    struct DecidedTwice;
    impl PlacementPolicy for DecidedTwice {
        fn name(&self) -> &str {
            "decided-twice"
        }
        fn place(
            &self,
            graph: &StageGraph,
            _b: &[BackendCapabilities],
        ) -> Result<Vec<PlacementDecision>, PlanError> {
            let mut decisions: Vec<_> = graph
                .stages()
                .iter()
                .map(|s| PlacementDecision::new(s.id, Placement::Preserve))
                .collect();
            decisions.push(decisions[0].clone());
            Ok(decisions)
        }
    }
    assert!(matches!(
        PlacementPlanner::new(DecidedTwice).dry_run(&discovered_graph()),
        Err(PlanError::InvalidPlacement { .. })
    ));

    struct Unknown;
    impl PlacementPolicy for Unknown {
        fn name(&self) -> &str {
            "unknown"
        }
        fn place(
            &self,
            graph: &StageGraph,
            _b: &[BackendCapabilities],
        ) -> Result<Vec<PlacementDecision>, PlanError> {
            let ghost: StageId = serde_json::from_str("42").unwrap();
            Ok(vec![PlacementDecision::new(ghost, Placement::Ann)]
                .into_iter()
                .chain(
                    graph
                        .stages()
                        .iter()
                        .map(|s| PlacementDecision::new(s.id, Placement::Preserve)),
                )
                .collect())
        }
    }
    assert!(matches!(
        PlacementPlanner::new(Unknown).dry_run(&discovered_graph()),
        Err(PlanError::InvalidPlacement { .. })
    ));
}

#[test]
fn dry_run_output_is_deterministic_and_inspectable() {
    let graph = discovered_graph();
    let planner = || {
        PlacementPlanner::new(TierAdapter)
            .offer(BackendCapabilities::ann("dense-host"))
            .offer(BackendCapabilities::adapter("boundary"))
            .offer(BackendCapabilities::snn("spike-rt"))
    };
    let a = planner().dry_run(&graph).unwrap();
    let b = planner().dry_run(&graph).unwrap();
    assert_eq!(a, b);
    assert_eq!(
        serde_json::to_string(&a).unwrap(),
        serde_json::to_string(&b).unwrap()
    );
    assert_eq!(a.describe(), b.describe());

    // The report carries the compiled HybridExecutionPlan.
    let text = a.describe();
    assert!(text.contains("policy 'tier-adapter'"), "{text}");
    assert_eq!(a.plan().execution_order().len(), graph.stages().len());
    // …and the input graph was not mutated by the run.
    let pristine = discovered_graph();
    assert_eq!(
        graph.stages().iter().map(|s| s.domain).collect::<Vec<_>>(),
        pristine
            .stages()
            .iter()
            .map(|s| s.domain)
            .collect::<Vec<_>>()
    );
}
