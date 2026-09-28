// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration coverage for RM-1804: `HybridExecutionPlan` stage graph and the
//! `HybridNetwork` / `ReverseHybridPath` compatibility paths.

use hybrid_fusion::error::Result;
use hybrid_fusion::plan::{EdgeKind, ExecutionDomain, PlanError};
use hybrid_fusion::{
    DimSpec, ExpertRouteOutput, ExpertRouter, HybridConfig, HybridNetwork, NeuroModulators,
    PortSpec, ProjectionMode, ReverseHybridPath, SpikeActivity, SpikingNetwork, StageGraph,
    StageKind, Tensor, Transformer,
};

struct MockTransformer {
    dim: usize,
    max_seq: usize,
}

impl Transformer for MockTransformer {
    fn hidden_states(&self, token_ids: &[u32]) -> Tensor {
        let seq = token_ids.len();
        Tensor::from_vec(vec![0.1; seq * self.dim], &[seq, self.dim])
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
}

struct MockSnn {
    channels: usize,
}

impl SpikingNetwork for MockSnn {
    fn step(&mut self, _stimuli: &[f32], _modulators: &NeuroModulators) -> Result<Vec<usize>> {
        Ok(Vec::new())
    }
    fn num_channels(&self) -> usize {
        self.channels
    }
}

struct MockRouter;

impl ExpertRouter for MockRouter {
    fn num_experts(&self) -> usize {
        4
    }
    fn top_k(&self) -> usize {
        2
    }
    fn route(&mut self, _embedding: &[f32]) -> Result<ExpertRouteOutput> {
        Ok(ExpertRouteOutput {
            expert_weights: vec![0.25; 4],
            selected_experts: vec![0, 1],
            routing_entropy: None,
        })
    }
}

#[test]
fn hybrid_network_execution_plan_matches_forward_flow() {
    let cfg = HybridConfig::tiny();
    let t = MockTransformer {
        dim: cfg.transformer.dim,
        max_seq: cfg.transformer.max_seq_len,
    };
    let s = MockSnn {
        channels: cfg.snn_input_channels,
    };
    let net = HybridNetwork::try_new(t, s, cfg.clone()).unwrap();
    let plan = net.execution_plan().unwrap();

    // Same three stages, in order, with config-derived contracts.
    assert_eq!(plan.execution_order().len(), 3);
    let transformer = plan.stage_by_name("ann.transformer").unwrap();
    assert_eq!(transformer.kind, StageKind::Transformer);
    let snn = plan.stage_by_name("snn.step").unwrap();
    assert_eq!(snn.input, PortSpec::f32_exact(&[cfg.snn_input_channels]));
    // Deterministic serialization round-trip.
    let json = plan.to_json().unwrap();
    assert_eq!(
        hybrid_fusion::HybridExecutionPlan::from_json(&json).unwrap(),
        plan
    );
}

#[test]
fn reverse_path_execution_plan_without_new_host() {
    let mut path = ReverseHybridPath::new(ProjectionMode::RateSum, 16, 32, MockRouter).unwrap();
    let plan = path.execution_plan().unwrap();
    assert_eq!(
        plan.stage_by_name("moe.router").unwrap().kind,
        StageKind::MoeRouter
    );
    // Still runnable through the existing host — plan is descriptive.
    let act = SpikeActivity::from_fired(&[1], 16).unwrap();
    path.forward_activity(&act).unwrap();
}

#[test]
fn cyclic_forward_graph_fails_with_structured_error() {
    let mut g = StageGraph::new();
    let a = g.add_stage("a", StageKind::DenseMlp, PortSpec::any(), PortSpec::any());
    let b = g.add_stage("b", StageKind::DenseMlp, PortSpec::any(), PortSpec::any());
    g.connect(a, b);
    g.connect(b, a);
    let err = g.compile().unwrap_err();
    assert!(matches!(err, PlanError::Cyclic { .. }));
    // Converts into the crate-wide error type.
    let herr: hybrid_fusion::HybridError = err.into();
    assert!(herr.to_string().contains("execution plan error"));
}

#[test]
fn feedback_loop_compiles_when_permitted() {
    let mut g = StageGraph::new();
    let ann = g.add_stage(
        "ann",
        StageKind::Transformer,
        PortSpec::any(),
        PortSpec::f32(vec![DimSpec::Fixed(8)]),
    );
    let snn = g.add_stage(
        "snn",
        StageKind::SpikingBlock,
        PortSpec::f32(vec![DimSpec::Fixed(8)]),
        PortSpec::any(),
    );
    g.allow_feedback(true);
    g.connect(ann, snn);
    g.connect_feedback(snn, ann);
    let plan = g.compile().unwrap();
    assert_eq!(
        plan.edges()
            .iter()
            .filter(|e| e.kind == EdgeKind::Feedback)
            .count(),
        1
    );
    assert_eq!(plan.resolved_domain(snn), Some(ExecutionDomain::Snn));
}
