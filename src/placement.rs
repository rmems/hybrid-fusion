// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generic placement policy and dry-run [`HybridExecutionPlan`] planning
//! (issue #24 / RM-330).
//!
//! A [`PlacementPolicy`] assigns every stage of a discovered [`StageGraph`]
//! to a [`Placement`] — dense ANN host, spiking runtime, the ANN↔SNN
//! boundary, or "preserved" on its discovered assignment. The policy is the
//! only pluggable piece: [`PlacementPlanner::dry_run`] collects the policy's
//! [`PlacementDecision`]s, validates that they form a total assignment over
//! the graph, pins the mapped [`ExecutionDomain`] on each stage, compiles the
//! graph through the full [`StageGraph::compile`] validation suite, and checks
//! offered [`BackendCapabilities`] cover every resolved domain. The result is
//! a deterministic, inspectable [`PlacementReport`] whose payload is a
//! validated [`HybridExecutionPlan`].
//!
//! Placement is a **policy-level** vocabulary, deliberately distinct from
//! [`ExecutionDomain`]: `Ann` / `Snn` pin the matching execution domain,
//! `Hybrid` pins the boundary [`ExecutionDomain::Adapter`] domain, and
//! `Preserve` leaves the stage's existing assignment untouched (an
//! [`ExecutionDomain::Auto`] stage still resolves to its kind's default).
//! Whether a pin is legal is decided by
//! [`StageKind::allowed_domains`](crate::StageKind::allowed_domains) at
//! compile time — an `Embedding` stage placed `Snn` fails with
//! [`PlanError::InvalidDomain`], not a silent re-route.
//!
//! Dry-run planning is file-free and weight-free by construction: policies
//! see names, kinds, port contracts, and capability reports — never tensors,
//! payloads, or paths. Model-family vocabularies (for example the
//! grok-ozempic precision tiers, adapted per RM-1817) ride
//! [`PlacementDecision`]'s `attrs` and merge into the stage's `attrs`; the
//! core enums never name them.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::capabilities::BackendCapabilities;
use crate::plan::{EdgeKind, ExecutionDomain, HybridExecutionPlan, PlanError, StageGraph, StageId};

// ── Placement vocabulary ────────────────────────────────────────────────────

/// Policy-level placement for one discovered stage (issue #24).
///
/// This is the *decision* vocabulary a [`PlacementPolicy`] emits; it is not
/// the execution-domain enum. The mapping is:
///
/// | `Placement` | Effect on the stage |
/// |-------------|---------------------|
/// | [`Ann`](Self::Ann) | pin [`ExecutionDomain::Ann`] |
/// | [`Snn`](Self::Snn) | pin [`ExecutionDomain::Snn`] |
/// | [`Hybrid`](Self::Hybrid) | pin [`ExecutionDomain::Adapter`] (the ANN↔SNN boundary domain) |
/// | [`Preserve`](Self::Preserve) | pin nothing — the stage keeps its existing domain assignment |
///
/// A pin outside [`StageKind::allowed_domains`](crate::StageKind::allowed_domains)
/// fails compilation with [`PlanError::InvalidDomain`]: only `Adaptation` and
/// `Readout` stages may be placed `Hybrid`, and only `SpikingBlock` stages may
/// be placed `Snn`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Placement {
    /// Run the stage on the dense ANN side.
    Ann,
    /// Migrate the stage to the spiking runtime.
    Snn,
    /// Place the stage on the ANN↔SNN boundary (`ExecutionDomain::Adapter`):
    /// it consumes ANN-side activations and emits SNN-side stimuli (or the
    /// reverse), which only boundary-capable stage kinds support.
    Hybrid,
    /// Keep the stage's discovered assignment: an `Auto` stage resolves to
    /// its kind's default domain, an already-pinned stage keeps its pin.
    /// The policy chooses not to re-place the stage.
    Preserve,
}

impl Placement {
    /// [`ExecutionDomain`] this placement pins on the stage, or `None` when
    /// the stage's existing assignment is kept ([`Preserve`](Self::Preserve)).
    pub fn domain_pin(self) -> Option<ExecutionDomain> {
        match self {
            Self::Ann => Some(ExecutionDomain::Ann),
            Self::Snn => Some(ExecutionDomain::Snn),
            Self::Hybrid => Some(ExecutionDomain::Adapter),
            Self::Preserve => None,
        }
    }
}

// ── Decisions ───────────────────────────────────────────────────────────────

/// A policy's placement assignment for one stage, plus policy metadata.
///
/// `attrs` carries policy-specific vocabulary (precision tiers, family tags,
/// experiment labels) without touching the core enums. `dry_run` merges the
/// map into the stage's `attrs` verbatim, so policies should namespace their
/// keys; a key that would overwrite an existing stage attribute fails with
/// [`PlanError::InvalidPlacement`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacementDecision {
    /// Stage this decision applies to.
    pub stage: StageId,
    /// Assigned placement.
    pub placement: Placement,
    /// Policy-specific metadata merged into `Stage::attrs`. Sorted map, so
    /// reports serialize deterministically.
    pub attrs: BTreeMap<String, String>,
}

impl PlacementDecision {
    /// Decision assigning `placement` to `stage` with no metadata.
    pub fn new(stage: StageId, placement: Placement) -> Self {
        Self {
            stage,
            placement,
            attrs: BTreeMap::new(),
        }
    }

    /// Attach a policy metadata pair (namespaced keys recommended).
    pub fn with_attr(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attrs.insert(key.into(), value.into());
        self
    }
}

// ── Policy contract ─────────────────────────────────────────────────────────

/// Pluggable policy that assigns a [`Placement`] to every stage of a
/// discovered graph (issue #24).
///
/// `place` is a pure decision pass over generic stage concepts — stage
/// names, [`StageKind`](crate::StageKind)s, [`PortSpec`](crate::PortSpec)
/// contracts, edges, and the offered backend reports. The contract gives the
/// policy no access to model weights or checkpoint files; keeping it pure is
/// what makes the dry run weight-free.
///
/// The returned set must cover every stage in `graph` exactly once.
/// [`PlacementPlanner::dry_run`] rejects unknown, duplicate, and missing
/// decisions with [`PlanError::InvalidPlacement`] before it applies any of
/// them, so a policy cannot produce a partially-placed plan.
///
/// Implementations live outside this crate. Research crates adapt their own
/// policy vocabularies into [`PlacementDecision`]s — for example
/// grok-ozempic maps its precision tiers onto `Preserve` / `Ann` /
/// `Hybrid` / `Snn` per RM-1817 — without those vocabularies becoming
/// hybrid-fusion types.
pub trait PlacementPolicy {
    /// Stable identity recorded on the [`PlacementReport`].
    fn name(&self) -> &str;

    /// Decide a placement for every stage in `graph`.
    ///
    /// `backends` is the offered, side-effect-free capability reports the
    /// planner was built with — the same [`BackendCapabilities`] vocabulary
    /// [`crate::CapabilityNegotiation`] consumes. A policy may decline a
    /// placement no backend can run (for example, refuse `Snn` when no
    /// SNN-domain report is offered) or leave detection to `dry_run`'s
    /// domain-coverage validation, which fails with
    /// [`PlanError::UnsupportedBackend`].
    ///
    /// # Errors
    ///
    /// A policy rejects the graph itself with a [`PlanError`]
    /// ([`PlanError::InvalidParameters`] fits policy-specific reasons).
    /// Decision-set validation belongs to `dry_run`, not to the policy.
    fn place(
        &self,
        graph: &StageGraph,
        backends: &[BackendCapabilities],
    ) -> Result<Vec<PlacementDecision>, PlanError>;
}

// ── Dry-run planner ─────────────────────────────────────────────────────────

/// Dry-run planning host: applies a [`PlacementPolicy`] to a [`StageGraph`]
/// and compiles the annotated graph into a validated [`HybridExecutionPlan`].
///
/// Offering a backend registers its report for **domain-coverage**
/// validation: after compile, every stage's resolved domain must be
/// advertised by at least one offered backend, else `dry_run` fails with
/// [`PlanError::UnsupportedBackend`]. This is feasibility, not selection —
/// choosing which compatible backend runs a stage remains
/// [`crate::CapabilityNegotiation`]'s job on the returned plan. With no
/// backends offered, `dry_run` produces a structural report only (domain
/// coverage is skipped).
///
/// Nothing here loads, mutates, or executes model weights: the input is a
/// stage graph, the output is a plan plus inspectable decisions.
#[derive(Debug, Clone)]
pub struct PlacementPlanner<P> {
    policy: P,
    backends: Vec<BackendCapabilities>,
}

impl<P: PlacementPolicy> PlacementPlanner<P> {
    /// Planner over `policy` with no backend reports offered.
    pub fn new(policy: P) -> Self {
        Self {
            policy,
            backends: Vec::new(),
        }
    }

    /// Offer a backend capability report for domain-coverage validation.
    /// Reports are read, never executed.
    ///
    /// Must use the returned planner: `offer` consumes `self`, and a discarded
    /// result leaves the original planner with no backends, which skips
    /// domain-coverage validation.
    #[must_use]
    pub fn offer(mut self, caps: BackendCapabilities) -> Self {
        self.backends.push(caps);
        self
    }

    /// Run the policy over `graph` and compile the placed plan.
    ///
    /// `graph` is not mutated: decisions are applied to a clone, so a failed
    /// dry run leaves the caller's builder untouched.
    ///
    /// # Errors
    ///
    /// - [`PlanError::InvalidPlacement`] if the policy's decision set does not
    ///   cover the graph exactly once, or a decision attribute would
    ///   overwrite an existing stage attribute.
    /// - [`PlanError::InvalidDomain`] (via [`StageGraph::compile`]) if a
    ///   placement pins a stage outside its kind's allowed domains.
    /// - Any other [`PlanError`] the graph's structural validation raises, or
    ///   the policy itself returns.
    /// - [`PlanError::UnsupportedBackend`] if an offered backend set is
    ///   non-empty and no report advertises a stage's resolved domain.
    pub fn dry_run(&self, graph: &StageGraph) -> Result<PlacementReport, PlanError> {
        let decisions = self.policy.place(graph, &self.backends)?;
        let placed = apply_decisions(graph, &decisions)?;
        let plan = placed.compile()?;
        self.check_domain_coverage(&plan)?;
        Ok(PlacementReport {
            policy: self.policy.name().to_string(),
            plan,
            decisions: normalize(decisions),
        })
    }

    /// [`dry_run`](Self::dry_run) over an already-compiled plan.
    ///
    /// The plan's stages (including existing domain pins and attributes) and
    /// edges are rebuilt into a [`StageGraph`] and re-placed; an all-
    /// [`Placement::Preserve`] policy therefore reproduces the input plan.
    /// Useful for re-placing a canonical plan such as
    /// [`HybridExecutionPlan::from_topology`].
    ///
    /// # Errors
    ///
    /// Same as [`dry_run`](Self::dry_run).
    pub fn dry_run_plan(&self, plan: &HybridExecutionPlan) -> Result<PlacementReport, PlanError> {
        self.dry_run(&graph_from_plan(plan))
    }

    /// Every stage's resolved domain must be advertised by at least one
    /// offered backend. Skipped when no backends were offered.
    fn check_domain_coverage(&self, plan: &HybridExecutionPlan) -> Result<(), PlanError> {
        if self.backends.is_empty() {
            return Ok(());
        }
        for stage in plan.stages() {
            let domain = plan
                .resolved_domain(stage.id)
                .expect("resolved domain exists for a known stage");
            if !self
                .backends
                .iter()
                .any(|caps| caps.supports_domain(domain))
            {
                return Err(PlanError::UnsupportedBackend {
                    stage: stage.id,
                    domain,
                    reason: format!(
                        "no offered backend advertises domain {domain:?} for stage '{}'",
                        stage.name
                    ),
                });
            }
        }
        Ok(())
    }
}

/// Validate `decisions` against `graph` and return the annotated graph clone.
///
/// A valid set decides every stage exactly once and never overwrites an
/// existing stage attribute. Failures are [`PlanError::InvalidPlacement`]
/// with the offending stages named in deterministic [`StageId`] order.
fn apply_decisions(
    graph: &StageGraph,
    decisions: &[PlacementDecision],
) -> Result<StageGraph, PlanError> {
    // Validate and apply in StageId order so the first error is deterministic
    // even when the policy returns decisions in a different order.
    let decisions = normalize(decisions.to_vec());
    let known: BTreeSet<StageId> = graph.stages().iter().map(|s| s.id).collect();
    let mut decided = BTreeSet::new();
    for d in &decisions {
        if !known.contains(&d.stage) {
            return Err(PlanError::InvalidPlacement {
                reason: format!(
                    "decision names stage {}, which is not in the graph",
                    d.stage
                ),
            });
        }
        if !decided.insert(d.stage) {
            return Err(PlanError::InvalidPlacement {
                reason: format!("stage {} has more than one decision", d.stage),
            });
        }
    }
    let missing: Vec<StageId> = known.difference(&decided).copied().collect();
    if !missing.is_empty() {
        return Err(PlanError::InvalidPlacement {
            reason: format!("no decision for stage(s) {missing:?}"),
        });
    }

    let mut placed = graph.clone();
    for d in decisions {
        if let Some(domain) = d.placement.domain_pin() {
            placed.set_domain(d.stage, domain);
        }
        let occupied = d
            .attrs
            .keys()
            .find(|key| {
                placed
                    .stages()
                    .iter()
                    .find(|s| s.id == d.stage)
                    .expect("stage validated above")
                    .attrs
                    .contains_key(*key)
            })
            .cloned();
        if let Some(key) = occupied {
            return Err(PlanError::InvalidPlacement {
                reason: format!(
                    "decision for stage {} would overwrite stage attribute '{key}'",
                    d.stage
                ),
            });
        }
        for (key, value) in &d.attrs {
            placed.set_attr(d.stage, key.clone(), value.clone());
        }
    }
    Ok(placed)
}

/// Rebuild a [`StageGraph`] from a compiled plan, preserving stage domains
/// and attributes. Sequential IDs and edge kinds are reproduced exactly, so
/// recompilation is a no-op when a policy re-places nothing.
fn graph_from_plan(plan: &HybridExecutionPlan) -> StageGraph {
    let mut graph = StageGraph::new();
    graph.allow_feedback(plan.allows_feedback());
    for stage in plan.stages() {
        let id = graph.add_stage(
            stage.name.clone(),
            stage.kind,
            stage.input.clone(),
            stage.output.clone(),
        );
        debug_assert_eq!(id, stage.id);
        graph.set_domain(id, stage.domain);
        for (key, value) in &stage.attrs {
            graph.set_attr(id, key.clone(), value.clone());
        }
    }
    for edge in plan.edges() {
        match edge.kind {
            EdgeKind::Forward => {
                graph.connect(edge.from, edge.to);
            }
            EdgeKind::Feedback => {
                graph.connect_feedback(edge.from, edge.to);
            }
        }
    }
    graph
}

/// Decisions in deterministic [`StageId`] order, regardless of the order the
/// policy returned them.
fn normalize(mut decisions: Vec<PlacementDecision>) -> Vec<PlacementDecision> {
    decisions.sort_by_key(|d| d.stage);
    decisions
}

// ── Report ──────────────────────────────────────────────────────────────────

/// Outcome of a [`PlacementPlanner::dry_run`]: the validated plan plus the
/// normalized decisions that produced it (issue #24).
///
/// The report is deterministic — same graph, policy, and backend offers
/// serialize and render identically — and inspectable: [`decisions`](Self::decisions)
/// lists every stage's [`Placement`] in [`StageId`] order and
/// [`plan`](Self::plan) is the compiled [`HybridExecutionPlan`].
///
/// `Deserialize` is deliberately not derived, matching
/// [`HybridExecutionPlan`]: plans re-enter through
/// [`HybridExecutionPlan::from_json`], which revalidates.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PlacementReport {
    /// [`PlacementPolicy::name`] of the policy that produced this report.
    policy: String,
    /// Compiled, validated execution plan.
    plan: HybridExecutionPlan,
    /// Applied decisions in [`StageId`] order.
    decisions: Vec<PlacementDecision>,
}

impl PlacementReport {
    /// Policy identity.
    pub fn policy(&self) -> &str {
        &self.policy
    }

    /// The dry-run's compiled [`HybridExecutionPlan`].
    pub fn plan(&self) -> &HybridExecutionPlan {
        &self.plan
    }

    /// Every applied decision, in [`StageId`] order.
    pub fn decisions(&self) -> &[PlacementDecision] {
        &self.decisions
    }

    /// The decision applied to `stage`, if any.
    pub fn decision_for(&self, stage: StageId) -> Option<&PlacementDecision> {
        self.decisions.iter().find(|d| d.stage == stage)
    }

    /// Deterministic multi-line rendering: policy identity plus each stage's
    /// placement and resolved domain, in [`StageId`] order. Suitable for
    /// review diffs — same dry run, same text.
    pub fn describe(&self) -> String {
        let mut out = format!("PlacementReport {{\n  policy '{}'\n", self.policy);
        for d in &self.decisions {
            let stage = self.plan.stage(d.stage).expect("decision covers stages");
            let domain = self
                .plan
                .resolved_domain(d.stage)
                .expect("resolved domain exists for a known stage");
            let attrs = if d.attrs.is_empty() {
                String::new()
            } else {
                format!(" attrs={:?}", d.attrs)
            };
            out.push_str(&format!(
                "  {} '{}' kind={:?} placement={:?} domain={domain:?}{attrs}\n",
                stage.id, stage.name, stage.kind, d.placement
            ));
        }
        out.push('}');
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{PortSpec, StageKind};

    struct AllPreserve;

    impl PlacementPolicy for AllPreserve {
        fn name(&self) -> &str {
            "all-preserve"
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

    struct Uniform(Placement);

    impl PlacementPolicy for Uniform {
        fn name(&self) -> &str {
            "uniform"
        }

        fn place(
            &self,
            graph: &StageGraph,
            _backends: &[BackendCapabilities],
        ) -> Result<Vec<PlacementDecision>, PlanError> {
            Ok(graph
                .stages()
                .iter()
                .map(|s| PlacementDecision::new(s.id, self.0))
                .collect())
        }
    }

    fn linear_graph() -> StageGraph {
        let mut g = StageGraph::new();
        let a = g.add_stage(
            "embed",
            StageKind::Embedding,
            PortSpec::any(),
            PortSpec::f32_exact(&[8]),
        );
        let b = g.add_stage(
            "adapt",
            StageKind::Adaptation,
            PortSpec::f32_exact(&[8]),
            PortSpec::f32_exact(&[4]),
        );
        let c = g.add_stage(
            "snn",
            StageKind::SpikingBlock,
            PortSpec::f32_exact(&[4]),
            PortSpec::any(),
        );
        g.connect(a, b);
        g.connect(b, c);
        g
    }

    #[test]
    fn preserve_everything_reproduces_the_default_plan() {
        let graph = linear_graph();
        let report = PlacementPlanner::new(AllPreserve).dry_run(&graph).unwrap();
        assert_eq!(report.plan(), &graph.compile().unwrap());
        assert_eq!(report.policy(), "all-preserve");
        assert_eq!(report.decisions().len(), 3);
        let adapt = graph.stages()[1].id;
        assert_eq!(
            report.plan().resolved_domain(adapt),
            Some(ExecutionDomain::Adapter)
        );
    }

    #[test]
    fn ann_pin_and_hybrid_pin_apply_to_allowed_kinds() {
        let graph = linear_graph();
        let err = PlacementPlanner::new(Uniform(Placement::Ann))
            .dry_run(&graph)
            .unwrap_err();
        // SpikingBlock cannot run on the ANN host.
        assert!(matches!(err, PlanError::InvalidDomain { .. }));

        struct AdaptHybrid;
        impl PlacementPolicy for AdaptHybrid {
            fn name(&self) -> &str {
                "adapt-hybrid"
            }
            fn place(
                &self,
                graph: &StageGraph,
                _b: &[BackendCapabilities],
            ) -> Result<Vec<PlacementDecision>, PlanError> {
                Ok(graph
                    .stages()
                    .iter()
                    .map(|s| {
                        PlacementDecision::new(
                            s.id,
                            if s.name == "adapt" {
                                Placement::Hybrid
                            } else {
                                Placement::Preserve
                            },
                        )
                    })
                    .collect())
            }
        }
        let report = PlacementPlanner::new(AdaptHybrid).dry_run(&graph).unwrap();
        let adapt = graph.stages()[1].id;
        assert_eq!(
            report.plan().resolved_domain(adapt),
            Some(ExecutionDomain::Adapter)
        );
    }

    #[test]
    fn unsupported_domain_pin_fails_at_compile() {
        // Embedding allows only Ann; a Snn placement must fail, not re-route.
        let err = PlacementPlanner::new(Uniform(Placement::Snn))
            .dry_run(&linear_graph())
            .unwrap_err();
        match err {
            PlanError::InvalidDomain { domain, .. } => {
                assert_eq!(domain, ExecutionDomain::Snn)
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn decision_set_must_cover_every_stage_once() {
        struct Partial;
        impl PlacementPolicy for Partial {
            fn name(&self) -> &str {
                "partial"
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
        let err = PlacementPlanner::new(Partial)
            .dry_run(&linear_graph())
            .unwrap_err();
        assert!(matches!(err, PlanError::InvalidPlacement { .. }), "{err:?}");
    }

    #[test]
    fn unknown_and_duplicate_decisions_fail() {
        struct Bad;
        impl PlacementPolicy for Bad {
            fn name(&self) -> &str {
                "bad"
            }
            fn place(
                &self,
                _g: &StageGraph,
                _b: &[BackendCapabilities],
            ) -> Result<Vec<PlacementDecision>, PlanError> {
                let unknown: StageId = serde_json::from_str("99").unwrap();
                Ok(vec![PlacementDecision::new(unknown, Placement::Ann)])
            }
        }
        assert!(matches!(
            PlacementPlanner::new(Bad).dry_run(&linear_graph()),
            Err(PlanError::InvalidPlacement { .. })
        ));

        struct Dup;
        impl PlacementPolicy for Dup {
            fn name(&self) -> &str {
                "dup"
            }
            fn place(
                &self,
                graph: &StageGraph,
                _b: &[BackendCapabilities],
            ) -> Result<Vec<PlacementDecision>, PlanError> {
                let mut v: Vec<_> = graph
                    .stages()
                    .iter()
                    .map(|s| PlacementDecision::new(s.id, Placement::Preserve))
                    .collect();
                v.push(v[0].clone());
                Ok(v)
            }
        }
        assert!(matches!(
            PlacementPlanner::new(Dup).dry_run(&linear_graph()),
            Err(PlanError::InvalidPlacement { .. })
        ));
    }

    #[test]
    fn domain_coverage_uses_offered_backends() {
        let graph = linear_graph();
        // An Ann-only backend cannot cover a spiking stage.
        let err = PlacementPlanner::new(AllPreserve)
            .offer(BackendCapabilities::ann("dense"))
            .dry_run(&graph)
            .unwrap_err();
        assert!(
            matches!(err, PlanError::UnsupportedBackend { .. }),
            "{err:?}"
        );

        let report = PlacementPlanner::new(AllPreserve)
            .offer(BackendCapabilities::ann("dense"))
            .offer(BackendCapabilities::adapter("boundary"))
            .offer(BackendCapabilities::snn("spikes"))
            .dry_run(&graph)
            .unwrap();
        assert_eq!(report.decisions().len(), 3);
    }

    #[test]
    fn decision_attrs_merge_into_stage_attrs_without_overwrite() {
        struct Tagged;
        impl PlacementPolicy for Tagged {
            fn name(&self) -> &str {
                "tagged"
            }
            fn place(
                &self,
                graph: &StageGraph,
                _b: &[BackendCapabilities],
            ) -> Result<Vec<PlacementDecision>, PlanError> {
                Ok(graph
                    .stages()
                    .iter()
                    .map(|s| {
                        PlacementDecision::new(s.id, Placement::Preserve)
                            .with_attr("policy.tier", "keep")
                    })
                    .collect())
            }
        }
        let report = PlacementPlanner::new(Tagged)
            .dry_run(&linear_graph())
            .unwrap();
        for stage in report.plan().stages() {
            assert_eq!(
                stage.attrs.get("policy.tier").map(String::as_str),
                Some("keep")
            );
        }

        struct Overwrite;
        impl PlacementPolicy for Overwrite {
            fn name(&self) -> &str {
                "overwrite"
            }
            fn place(
                &self,
                graph: &StageGraph,
                _b: &[BackendCapabilities],
            ) -> Result<Vec<PlacementDecision>, PlanError> {
                Ok(graph
                    .stages()
                    .iter()
                    .map(|s| {
                        PlacementDecision::new(s.id, Placement::Preserve).with_attr("role", "x")
                    })
                    .collect())
            }
        }
        let mut graph = linear_graph();
        let embed = graph.stages()[0].id;
        graph.set_attr(embed, "role", "embedding");
        assert!(matches!(
            PlacementPlanner::new(Overwrite).dry_run(&graph),
            Err(PlanError::InvalidPlacement { .. })
        ));
    }

    #[test]
    fn dry_run_is_deterministic_regardless_of_decision_order() {
        struct Reversed;
        impl PlacementPolicy for Reversed {
            fn name(&self) -> &str {
                "reversed"
            }
            fn place(
                &self,
                graph: &StageGraph,
                _b: &[BackendCapabilities],
            ) -> Result<Vec<PlacementDecision>, PlanError> {
                Ok(graph
                    .stages()
                    .iter()
                    .rev()
                    .map(|s| PlacementDecision::new(s.id, Placement::Preserve))
                    .collect())
            }
        }
        let a = PlacementPlanner::new(AllPreserve)
            .dry_run(&linear_graph())
            .unwrap();
        let b = PlacementPlanner::new(Reversed)
            .dry_run(&linear_graph())
            .unwrap();
        assert_eq!(a.plan(), b.plan());
        assert_eq!(a.decisions(), b.decisions());
        // Same policy and input render byte-identically on a second run.
        let a2 = PlacementPlanner::new(AllPreserve)
            .dry_run(&linear_graph())
            .unwrap();
        assert_eq!(a.describe(), a2.describe());
    }

    #[test]
    fn dry_run_plan_replaces_a_compiled_plan() {
        let plan = linear_graph().compile().unwrap();
        let report = PlacementPlanner::new(AllPreserve)
            .dry_run_plan(&plan)
            .unwrap();
        assert_eq!(report.plan(), &plan);
    }

    #[test]
    fn report_renders_inspectably() {
        let report = PlacementPlanner::new(AllPreserve)
            .dry_run(&linear_graph())
            .unwrap();
        let text = report.describe();
        assert!(text.contains("policy 'all-preserve'"), "{text}");
        assert!(text.contains("placement=Preserve"), "{text}");
        assert!(text.contains("domain=Snn"), "{text}");
        assert_eq!(
            report.decision_for(report.decisions()[0].stage),
            report.decisions().first()
        );
        assert!(
            serde_json::to_string(&report)
                .unwrap()
                .contains("all-preserve")
        );
    }
}
