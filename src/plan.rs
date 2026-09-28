// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed stage graph and validated [`HybridExecutionPlan`] (v0.4 execution
//! contract, RM-1804 / issue #40).
//!
//! This module is the structural description of a hybrid ANN ↔ SNN pipeline.
//! It owns **no numerical implementation**: stages carry names, kinds,
//! shape/dtype port contracts, and an execution-domain assignment; edges are
//! explicit [`EdgeKind::Forward`] or [`EdgeKind::Feedback`] connections.
//! [`StageGraph::compile`] validates the graph into a deterministic
//! [`HybridExecutionPlan`] — deterministic stage IDs, deterministic
//! topological order, deterministic serialization — while the existing
//! [`crate::HybridNetwork`] / [`crate::ReverseHybridPath`] hosts keep their
//! numerical semantics unchanged.
//!
//! Capability negotiation (RM-1805 `BackendCapabilities`) plugs in later via
//! the per-stage [`ExecutionDomain`] hooks; this module deliberately does not
//! model device capabilities, checkpoint parsing, neuron dynamics, or
//! model-family policy.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::types::{Dtype, HybridConfig, ProjectionMode};

// ── Identifiers ─────────────────────────────────────────────────────────────

/// Deterministic stage identifier: assigned in insertion order (`s0`, `s1`, …)
/// and stable across serialization round-trips.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StageId(u32);

impl StageId {
    /// Raw numeric index (insertion order).
    pub fn index(self) -> u32 {
        self.0
    }
}

impl fmt::Display for StageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "s{}", self.0)
    }
}

// ── Stage taxonomy ──────────────────────────────────────────────────────────

/// Semantic role of a stage in the graph.
///
/// Each kind maps to a default [`ExecutionDomain`] and an allowed-domain set
/// used by [`StageGraph::compile`]; the mapping is intentionally coarse — it
/// places the stage on the side of the ANN/SNN boundary it belongs to, not on
/// a specific device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageKind {
    /// Token / position embedding lookup.
    Embedding,
    /// Attention block (self- or cross-attention).
    Attention,
    /// Dense feed-forward / MLP block.
    DenseMlp,
    /// Monolithic transformer stack (embedding + attention + MLP treated as a
    /// single stage, matching the current [`crate::Transformer`] trait).
    Transformer,
    /// MoE gating / router stage.
    MoeRouter,
    /// MoE expert evaluation stage.
    MoeExperts,
    /// ANN ↔ SNN projection / adaptation stage (e.g. `project_spike_activity`
    /// or `embed_to_stimuli_with_width`).
    Adaptation,
    /// Spiking network block (one or more SNN timesteps).
    SpikingBlock,
    /// Readout / output projection stage.
    Readout,
}

impl StageKind {
    /// Domain a stage runs in when [`ExecutionDomain::Auto`] is assigned.
    pub fn default_domain(self) -> ExecutionDomain {
        match self {
            Self::Embedding
            | Self::Attention
            | Self::DenseMlp
            | Self::Transformer
            | Self::Readout => ExecutionDomain::Ann,
            Self::MoeRouter | Self::MoeExperts => ExecutionDomain::Moe,
            Self::Adaptation => ExecutionDomain::Adapter,
            Self::SpikingBlock => ExecutionDomain::Snn,
        }
    }

    /// Domains this kind may be explicitly assigned to.
    ///
    /// A stage pinned outside its allowed set fails compilation with
    /// [`PlanError::InvalidDomain`]. [`ExecutionDomain::Auto`] is always
    /// accepted and resolves to [`Self::default_domain`].
    pub fn allowed_domains(self) -> &'static [ExecutionDomain] {
        match self {
            Self::Embedding | Self::Attention | Self::DenseMlp | Self::Transformer => {
                &[ExecutionDomain::Ann]
            }
            Self::MoeRouter | Self::MoeExperts => &[ExecutionDomain::Moe, ExecutionDomain::Ann],
            Self::Adaptation => &[ExecutionDomain::Adapter, ExecutionDomain::Ann],
            Self::SpikingBlock => &[ExecutionDomain::Snn],
            Self::Readout => &[ExecutionDomain::Ann, ExecutionDomain::Adapter],
        }
    }
}

/// Execution-domain assignment for a stage.
///
/// This is the **assignment hook** RM-1804 exposes for later capability
/// negotiation (RM-1805): a planner may pin a stage to a domain with
/// [`StageGraph::set_domain`], subject to [`StageKind::allowed_domains`].
/// [`Auto`](Self::Auto) defers to the kind's default domain at compile time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ExecutionDomain {
    /// Resolved from [`StageKind::default_domain`] during compilation.
    #[default]
    Auto,
    /// Dense ANN host compute (transformer, embedding, MLP, readout).
    Ann,
    /// Spiking runtime (SNN timestep execution).
    Snn,
    /// MoE routing / expert evaluation.
    Moe,
    /// ANN ↔ SNN boundary adapter (projection, pooling, encoding).
    Adapter,
}

// ── Port contracts ──────────────────────────────────────────────────────────

/// One dimension of a port's shape contract.
///
/// `Fixed` pins an extent; `Symbolic` binds a name that unifies across edges
/// of the same plan (e.g. `"seq"`); `Any` accepts any extent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DimSpec {
    /// Any extent is acceptable.
    Any,
    /// Exact extent; must be `> 0` (same zero-axis rule as [`crate::Tensor`]).
    Fixed(usize),
    /// Named extent unified across the plan; two different bindings of the
    /// same symbol are incompatible.
    Symbolic(String),
}

/// Shape/dtype contract on a stage input or output port.
///
/// `dtype: None` means "any dtype". `dims: []` means "rank unconstrained"
/// (wildcard) — it does **not** mean a rank-0 scalar.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PortSpec {
    /// Element dtype, if constrained.
    pub dtype: Option<Dtype>,
    /// Per-axis contract; empty = rank unconstrained.
    pub dims: Vec<DimSpec>,
}

impl PortSpec {
    /// Unconstrained port (any dtype, any rank). Use sparingly: explicit
    /// contracts catch malformed graphs earlier.
    pub fn any() -> Self {
        Self {
            dtype: None,
            dims: Vec::new(),
        }
    }

    /// `f32` port with the given per-axis contract.
    pub fn f32(dims: Vec<DimSpec>) -> Self {
        Self {
            dtype: Some(Dtype::F32),
            dims,
        }
    }

    /// `f32` port with all axes pinned to fixed extents.
    pub fn f32_exact(dims: &[usize]) -> Self {
        Self::f32(dims.iter().map(|&d| DimSpec::Fixed(d)).collect())
    }

    /// Test compatibility with a producer port under `bindings`, mutating
    /// `bindings` to record newly unified symbolic extents. Returns a
    /// human-readable reason on mismatch.
    fn compatible_with(
        producer: &PortSpec,
        consumer: &PortSpec,
        bindings: &mut BTreeMap<String, Option<usize>>,
    ) -> std::result::Result<(), String> {
        if let (Some(p), Some(c)) = (producer.dtype, consumer.dtype)
            && p != c
        {
            return Err(format!("dtype mismatch: {p:?} vs {c:?}"));
        }
        if !producer.dims.is_empty() && !consumer.dims.is_empty() {
            if producer.dims.len() != consumer.dims.len() {
                return Err(format!(
                    "rank mismatch: {} vs {}",
                    producer.dims.len(),
                    consumer.dims.len()
                ));
            }
            for (axis, (p, c)) in producer.dims.iter().zip(&consumer.dims).enumerate() {
                unify_dim(p, c, bindings).map_err(|r| format!("axis {axis}: {r}"))?;
            }
        }
        Ok(())
    }
}

fn unify_dim(
    producer: &DimSpec,
    consumer: &DimSpec,
    bindings: &mut BTreeMap<String, Option<usize>>,
) -> std::result::Result<(), String> {
    use DimSpec::{Any, Fixed, Symbolic};
    match (producer, consumer) {
        (Any, _) | (_, Any) => Ok(()),
        (Fixed(a), Fixed(b)) if a == b => Ok(()),
        (Fixed(a), Fixed(b)) => Err(format!("extent mismatch: {a} vs {b}")),
        (Symbolic(name), other) | (other, Symbolic(name)) => {
            let concrete = match other {
                Fixed(v) => Some(*v),
                Symbolic(_) => None,
                Any => unreachable!("handled above"),
            };
            match bindings.get(name) {
                Some(&bound) if bound == concrete => Ok(()),
                Some(&Some(bound)) => Err(format!(
                    "symbol '{name}' already bound to {bound}, cannot rebind"
                )),
                _ => {
                    bindings.insert(name.clone(), concrete);
                    Ok(())
                }
            }
        }
    }
}

// ── Edges ───────────────────────────────────────────────────────────────────

/// Direction/intent of a graph edge.
///
/// `Forward` edges must form a DAG and define the compiled topological order.
/// `Feedback` edges are explicit recurrent/reverse connections; they are
/// excluded from ordering but are still contract-checked, and require the
/// graph to opt in via [`StageGraph::allow_feedback`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeKind {
    /// Producer → consumer dataflow edge.
    Forward,
    /// Explicit recurrent / reverse-path edge (e.g. SNN activity fed back).
    Feedback,
}

/// A directed edge between two stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    /// Producer stage.
    pub from: StageId,
    /// Consumer stage.
    pub to: StageId,
    /// Forward dataflow or explicit feedback.
    pub kind: EdgeKind,
}

// ── Stages ──────────────────────────────────────────────────────────────────

/// One node of the stage graph: name, kind, port contracts, domain hook, and
/// free-form deterministic metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stage {
    /// Deterministic ID assigned at insertion.
    pub id: StageId,
    /// Human-readable name (must be unique within the graph).
    pub name: String,
    /// Semantic role.
    pub kind: StageKind,
    /// Domain assignment hook (`Auto` resolves at compile time).
    pub domain: ExecutionDomain,
    /// Input port contract; checked against every incoming edge's producer.
    pub input: PortSpec,
    /// Output port contract; checked against every outgoing edge's consumer.
    pub output: PortSpec,
    /// Deterministic metadata (sorted map), e.g. `projection_mode`.
    pub attrs: BTreeMap<String, String>,
}

// ── Errors ──────────────────────────────────────────────────────────────────

/// Structured validation failure for [`StageGraph::compile`].
///
/// Every variant carries the offending IDs/names so callers can pinpoint the
/// defect without parsing the message text.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum PlanError {
    /// The graph has no stages.
    #[error("empty stage graph")]
    Empty,

    /// Two stages share a name.
    #[error("duplicate stage name '{name}'")]
    DuplicateStageName { name: String },

    /// An edge references a stage ID not present in the graph.
    #[error("edge endpoint {stage} does not name a stage")]
    UnknownStage { stage: StageId },

    /// A stage is not reachable from any forward-edge source.
    #[error("stage {stage} ('{name}') is disconnected from the forward flow")]
    Disconnected { stage: StageId, name: String },

    /// Forward edges form a cycle; the listed stages participate in the
    /// unsorted remainder (in deterministic ID order).
    #[error("forward edges form a cycle involving stages {stages:?}")]
    Cyclic { stages: Vec<StageId> },

    /// A `Feedback` edge exists but the graph did not opt in via
    /// [`StageGraph::allow_feedback`].
    #[error("feedback edge {from} -> {to} requires StageGraph::allow_feedback(true)")]
    FeedbackNotPermitted { from: StageId, to: StageId },

    /// Producer output and consumer input contracts do not unify.
    #[error("edge {from} -> {to}: incompatible port contract: {reason}")]
    IncompatibleContract {
        from: StageId,
        to: StageId,
        reason: String,
    },

    /// A stage was pinned to a domain outside its kind's allowed set.
    #[error("stage {stage} ('{name}') of kind {kind:?} cannot run in domain {domain:?}")]
    InvalidDomain {
        stage: StageId,
        name: String,
        kind: StageKind,
        domain: ExecutionDomain,
    },

    /// A port contract is itself malformed (e.g. `Fixed(0)`).
    #[error("stage {stage} ('{name}') has invalid port contract: {reason}")]
    InvalidContract {
        stage: StageId,
        name: String,
        reason: String,
    },

    /// Serialization round-trip of a plan failed structural re-validation.
    #[error("serialized plan is structurally inconsistent: {0}")]
    Inconsistent(String),
}

// ── Graph builder ───────────────────────────────────────────────────────────

/// Builder for a typed stage graph. Stages are appended in deterministic
/// order; [`compile`](Self::compile) validates and freezes a
/// [`HybridExecutionPlan`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StageGraph {
    stages: Vec<Stage>,
    edges: Vec<Edge>,
    allow_feedback: bool,
}

impl StageGraph {
    /// Empty graph with no feedback opt-in.
    pub fn new() -> Self {
        Self::default()
    }

    /// Opt in (or out) of [`EdgeKind::Feedback`] edges. Default: off.
    pub fn allow_feedback(&mut self, allowed: bool) -> &mut Self {
        self.allow_feedback = allowed;
        self
    }

    /// Append a stage and return its deterministic [`StageId`].
    pub fn add_stage(
        &mut self,
        name: impl Into<String>,
        kind: StageKind,
        input: PortSpec,
        output: PortSpec,
    ) -> StageId {
        let id = StageId(self.stages.len() as u32);
        self.stages.push(Stage {
            id,
            name: name.into(),
            kind,
            domain: ExecutionDomain::Auto,
            input,
            output,
            attrs: BTreeMap::new(),
        });
        id
    }

    /// Pin a stage's execution domain (RM-1805 negotiation hook).
    ///
    /// Returns `false` if `id` is not a stage of this graph; domain legality
    /// itself is checked by [`compile`](Self::compile).
    pub fn set_domain(&mut self, id: StageId, domain: ExecutionDomain) -> bool {
        match self.stages.get_mut(id.index() as usize) {
            Some(stage) if stage.id == id => {
                stage.domain = domain;
                true
            }
            _ => false,
        }
    }

    /// Set a metadata attribute on a stage.
    ///
    /// Returns `false` if `id` is not a stage of this graph.
    pub fn set_attr(
        &mut self,
        id: StageId,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> bool {
        match self.stages.get_mut(id.index() as usize) {
            Some(stage) if stage.id == id => {
                stage.attrs.insert(key.into(), value.into());
                true
            }
            _ => false,
        }
    }

    /// Add a [`EdgeKind::Forward`] edge.
    pub fn connect(&mut self, from: StageId, to: StageId) -> &mut Self {
        self.edges.push(Edge {
            from,
            to,
            kind: EdgeKind::Forward,
        });
        self
    }

    /// Add an [`EdgeKind::Feedback`] edge (requires
    /// [`allow_feedback`](Self::allow_feedback) at compile time).
    pub fn connect_feedback(&mut self, from: StageId, to: StageId) -> &mut Self {
        self.edges.push(Edge {
            from,
            to,
            kind: EdgeKind::Feedback,
        });
        self
    }

    /// Stages in insertion order.
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// Edges in insertion order.
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Look up a stage by name.
    pub fn stage_by_name(&self, name: &str) -> Option<&Stage> {
        self.stages.iter().find(|s| s.name == name)
    }

    /// Validate and freeze into a [`HybridExecutionPlan`].
    ///
    /// Checks, in order: non-empty graph, unique names, known edge endpoints,
    /// feedback opt-in, domain legality, per-stage contract sanity, forward-DAG
    /// acyclicity (Kahn's algorithm with `StageId` ordering — deterministic),
    /// full forward reachability (no disconnected stages), and per-edge port
    /// contract unification (shared symbolic bindings).
    pub fn compile(&self) -> std::result::Result<HybridExecutionPlan, PlanError> {
        if self.stages.is_empty() {
            return Err(PlanError::Empty);
        }
        check_unique_names(&self.stages)?;
        self.check_endpoints()?;
        self.check_feedback_permitted()?;
        check_domains(&self.stages)?;
        check_contract_sanity(&self.stages)?;
        let order = self.forward_topo_order()?;
        self.check_reachability()?;
        self.check_edge_contracts()?;

        Ok(HybridExecutionPlan {
            stages: self.stages.clone(),
            edges: self.edges.clone(),
            order,
            allow_feedback: self.allow_feedback,
        })
    }

    fn check_endpoints(&self) -> std::result::Result<(), PlanError> {
        for e in &self.edges {
            for endpoint in [e.from, e.to] {
                if self.stage(endpoint).is_none() {
                    return Err(PlanError::UnknownStage { stage: endpoint });
                }
            }
        }
        Ok(())
    }

    fn stage(&self, id: StageId) -> Option<&Stage> {
        self.stages.get(id.index() as usize).filter(|s| s.id == id)
    }

    fn check_feedback_permitted(&self) -> std::result::Result<(), PlanError> {
        if self.allow_feedback {
            return Ok(());
        }
        if let Some(e) = self.edges.iter().find(|e| e.kind == EdgeKind::Feedback) {
            return Err(PlanError::FeedbackNotPermitted {
                from: e.from,
                to: e.to,
            });
        }
        Ok(())
    }

    /// Kahn's algorithm over forward edges only; ties break by `StageId` so the
    /// resulting order is deterministic.
    fn forward_topo_order(&self) -> std::result::Result<Vec<StageId>, PlanError> {
        let mut in_degree: BTreeMap<StageId, usize> =
            self.stages.iter().map(|s| (s.id, 0)).collect();
        let mut forward_adj: BTreeMap<StageId, Vec<StageId>> = BTreeMap::new();
        for e in self.edges.iter().filter(|e| e.kind == EdgeKind::Forward) {
            *in_degree.entry(e.to).or_insert(0) += 1;
            forward_adj.entry(e.from).or_default().push(e.to);
        }
        // BTreeSet gives deterministic smallest-ID-first extraction.
        let mut ready: BTreeSet<StageId> = in_degree
            .iter()
            .filter(|&(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();
        let mut order = Vec::with_capacity(self.stages.len());
        while let Some(&id) = ready.iter().next() {
            ready.remove(&id);
            order.push(id);
            if let Some(nexts) = forward_adj.get(&id) {
                for &n in nexts {
                    let d = in_degree.get_mut(&n).expect("endpoint checked");
                    *d -= 1;
                    if *d == 0 {
                        ready.insert(n);
                    }
                }
            }
        }
        if order.len() != self.stages.len() {
            let mut stages: Vec<StageId> = in_degree
                .iter()
                .filter(|&(_, &d)| d > 0)
                .map(|(&id, _)| id)
                .collect();
            stages.sort_unstable();
            return Err(PlanError::Cyclic { stages });
        }
        Ok(order)
    }

    /// The graph must be a single weakly-connected component over all edges:
    /// every stage reachable from `s0` treating edges as undirected. Orphan
    /// stages and second disconnected chains both fail.
    fn check_reachability(&self) -> std::result::Result<(), PlanError> {
        let mut seen: BTreeSet<StageId> = BTreeSet::from([self.stages[0].id]);
        let mut work: Vec<StageId> = vec![self.stages[0].id];
        while let Some(id) = work.pop() {
            for e in &self.edges {
                let next = if e.from == id {
                    Some(e.to)
                } else if e.to == id {
                    Some(e.from)
                } else {
                    None
                };
                if let Some(n) = next
                    && seen.insert(n)
                {
                    work.push(n);
                }
            }
        }
        if let Some(&stage) = self
            .stages
            .iter()
            .map(|s| &s.id)
            .find(|id| !seen.contains(id))
        {
            let name = self
                .stage(stage)
                .map(|s| s.name.clone())
                .unwrap_or_default();
            return Err(PlanError::Disconnected { stage, name });
        }
        Ok(())
    }

    fn check_edge_contracts(&self) -> std::result::Result<(), PlanError> {
        let mut bindings: BTreeMap<String, Option<usize>> = BTreeMap::new();
        for e in &self.edges {
            let producer = self.stage(e.from).expect("endpoint checked");
            let consumer = self.stage(e.to).expect("endpoint checked");
            PortSpec::compatible_with(&producer.output, &consumer.input, &mut bindings).map_err(
                |reason| PlanError::IncompatibleContract {
                    from: e.from,
                    to: e.to,
                    reason,
                },
            )?;
        }
        Ok(())
    }
}

fn check_unique_names(stages: &[Stage]) -> std::result::Result<(), PlanError> {
    let mut seen = BTreeSet::new();
    for s in stages {
        if !seen.insert(&s.name) {
            return Err(PlanError::DuplicateStageName {
                name: s.name.clone(),
            });
        }
    }
    Ok(())
}

fn check_domains(stages: &[Stage]) -> std::result::Result<(), PlanError> {
    for s in stages {
        if s.domain != ExecutionDomain::Auto && !s.kind.allowed_domains().contains(&s.domain) {
            return Err(PlanError::InvalidDomain {
                stage: s.id,
                name: s.name.clone(),
                kind: s.kind,
                domain: s.domain,
            });
        }
    }
    Ok(())
}

fn check_contract_sanity(stages: &[Stage]) -> std::result::Result<(), PlanError> {
    for s in stages {
        for port in [&s.input, &s.output] {
            for dim in &port.dims {
                if matches!(dim, DimSpec::Fixed(0)) {
                    return Err(PlanError::InvalidContract {
                        stage: s.id,
                        name: s.name.clone(),
                        reason: "Fixed(0) extent violates the >0 axis rule".into(),
                    });
                }
            }
        }
    }
    Ok(())
}

// ── Compiled plan ───────────────────────────────────────────────────────────

/// A validated, frozen stage graph.
///
/// `order` is the deterministic topological order over forward edges
/// (smallest `StageId` first among ready nodes). Feedback edges never appear
/// in `order`; they describe recurrence for the executor, not sequencing.
///
/// Serialize with [`to_json`](Self::to_json) / [`from_json`](Self::from_json);
/// field order and stage ordering are deterministic, so identical plans hash
/// and diff identically.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridExecutionPlan {
    stages: Vec<Stage>,
    edges: Vec<Edge>,
    order: Vec<StageId>,
    allow_feedback: bool,
}

impl HybridExecutionPlan {
    /// All stages in insertion (ID) order.
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// All edges (forward and feedback) in insertion order.
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Deterministic execution order over forward edges.
    pub fn execution_order(&self) -> &[StageId] {
        &self.order
    }

    /// Whether feedback edges are permitted in this plan.
    pub fn allows_feedback(&self) -> bool {
        self.allow_feedback
    }

    /// Look up a stage by [`StageId`].
    pub fn stage(&self, id: StageId) -> Option<&Stage> {
        self.stages.get(id.index() as usize).filter(|s| s.id == id)
    }

    /// Look up a stage by name.
    pub fn stage_by_name(&self, name: &str) -> Option<&Stage> {
        self.stages.iter().find(|s| s.name == name)
    }

    /// The stage's resolved execution domain (`Auto` is never returned).
    pub fn resolved_domain(&self, id: StageId) -> Option<ExecutionDomain> {
        self.stage(id).map(|s| {
            if s.domain == ExecutionDomain::Auto {
                s.kind.default_domain()
            } else {
                s.domain
            }
        })
    }

    /// Forward edges into `stage`, in insertion order.
    pub fn forward_inputs(&self, stage: StageId) -> Vec<&Edge> {
        self.edges
            .iter()
            .filter(|e| e.kind == EdgeKind::Forward && e.to == stage)
            .collect()
    }

    /// Feedback edges into `stage`, in insertion order.
    pub fn feedback_inputs(&self, stage: StageId) -> Vec<&Edge> {
        self.edges
            .iter()
            .filter(|e| e.kind == EdgeKind::Feedback && e.to == stage)
            .collect()
    }

    /// Deterministic JSON serialization (pretty-printed).
    pub fn to_json(&self) -> crate::error::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize and re-validate structural consistency (ID ↔ index
    /// correspondence and forward-DAG order coverage). Contract and domain
    /// checks are not re-run — a serialized plan was already validated.
    pub fn from_json(json: &str) -> std::result::Result<Self, PlanError> {
        let plan: Self =
            serde_json::from_str(json).map_err(|e| PlanError::Inconsistent(e.to_string()))?;
        for (i, s) in plan.stages.iter().enumerate() {
            if s.id.index() as usize != i {
                return Err(PlanError::Inconsistent(format!(
                    "stage index {i} carries non-sequential id {}",
                    s.id
                )));
            }
        }
        if plan.order.len() != plan.stages.len() {
            return Err(PlanError::Inconsistent(format!(
                "execution order has {} entries for {} stages",
                plan.order.len(),
                plan.stages.len()
            )));
        }
        let mut order_set: BTreeSet<StageId> = plan.order.iter().copied().collect();
        for s in &plan.stages {
            if !order_set.remove(&s.id) {
                return Err(PlanError::Inconsistent(format!(
                    "execution order omits stage {}",
                    s.id
                )));
            }
        }
        if !order_set.is_empty() {
            return Err(PlanError::Inconsistent(
                "execution order references unknown stages".into(),
            ));
        }
        // The stored order must be exactly the deterministic forward
        // topological order — this also rejects duplicates and orders that
        // violate forward-edge precedence.
        let graph = StageGraph {
            stages: plan.stages.clone(),
            edges: plan.edges.clone(),
            allow_feedback: plan.allow_feedback,
        };
        let expected = graph
            .forward_topo_order()
            .map_err(|e| PlanError::Inconsistent(e.to_string()))?;
        if expected != plan.order {
            return Err(PlanError::Inconsistent(
                "execution order is not the deterministic forward order".into(),
            ));
        }
        Ok(plan)
    }

    /// Deterministic multi-line debug rendering (stage order, edges, domains).
    ///
    /// Suitable for plan review diffs: same graph → same text.
    pub fn describe(&self) -> String {
        let mut out = String::from("HybridExecutionPlan {\n");
        for (i, &id) in self.order.iter().enumerate() {
            let s = self.stage(id).expect("order covers stages");
            let domain = self.resolved_domain(id).expect("stage exists");
            out.push_str(&format!(
                "  [{i}] {} '{}' kind={:?} domain={domain:?}\n",
                s.id, s.name, s.kind
            ));
        }
        for e in &self.edges {
            out.push_str(&format!("  edge {:?} {} -> {}\n", e.kind, e.from, e.to));
        }
        out.push('}');
        out
    }

    // ── Compatibility constructors (RM-1804) ────────────────────────────

    /// Plan describing the existing [`crate::HybridNetwork`] forward flow:
    /// `token_ids → Transformer → Adaptation(project/tanh) → SpikingBlock`.
    ///
    /// This is a **description** of the current semantics, not a replacement:
    /// numerical execution still lives in `HybridNetwork::forward`. Stage
    /// contracts are derived from `config` (`transformer.dim`,
    /// `snn_input_channels`).
    ///
    /// # Errors
    ///
    /// [`PlanError`] if `config` carries zero-valued dimensions (a
    /// `Fixed(0)` contract is invalid; note [`crate::HybridNetwork::new`]
    /// permits such configs without validation).
    pub fn from_hybrid_config(config: &HybridConfig) -> std::result::Result<Self, PlanError> {
        let mut g = StageGraph::new();
        let ann = g.add_stage(
            "ann.transformer",
            StageKind::Transformer,
            // token_ids: [seq] of u32
            PortSpec {
                dtype: Some(Dtype::U32),
                dims: vec![DimSpec::Symbolic("seq".into())],
            },
            // hidden states: [seq, dim] f32 (rank-1 [dim] is the pre-pooled
            // layout; the contract models the canonical rank-2 form)
            PortSpec::f32(vec![
                DimSpec::Symbolic("seq".into()),
                DimSpec::Fixed(config.transformer.dim),
            ]),
        );
        g.set_attr(ann, "role", "transformer.hidden_states");

        let adapt = g.add_stage(
            "adapt.project_stimuli",
            StageKind::Adaptation,
            PortSpec::f32(vec![
                DimSpec::Symbolic("seq".into()),
                DimSpec::Fixed(config.transformer.dim),
            ]),
            // bounded stimuli in [-1, 1] via projector tanh
            PortSpec::f32_exact(&[config.snn_input_channels]),
        );
        g.set_attr(adapt, "role", "projector::embed_to_stimuli_with_width");
        g.set_attr(adapt, "output_bounds", "[-1,1]");

        let snn = g.add_stage(
            "snn.step",
            StageKind::SpikingBlock,
            PortSpec::f32_exact(&[config.snn_input_channels]),
            // fired neuron indices (variable length)
            PortSpec {
                dtype: Some(Dtype::U64),
                dims: vec![DimSpec::Any],
            },
        );
        g.set_attr(snn, "role", "spiking_network.step");

        g.connect(ann, adapt);
        g.connect(adapt, snn);
        g.compile()
    }

    /// Plan describing the existing [`crate::ReverseHybridPath`] flow:
    /// `SpikeActivity → Adaptation(project) → MoeRouter → Readout`.
    ///
    /// Represents the reverse direction as an ordinary stage pipeline instead
    /// of a second unrelated host abstraction; `mode` is recorded as stage
    /// metadata.
    ///
    /// # Errors
    ///
    /// [`PlanError`] if `n_neurons` or `embed_dim` is `0` (invalid `Fixed(0)`
    /// contract).
    pub fn from_reverse_path(
        mode: ProjectionMode,
        n_neurons: usize,
        embed_dim: usize,
    ) -> std::result::Result<Self, PlanError> {
        let mut g = StageGraph::new();
        let activity = g.add_stage(
            "snn.activity",
            StageKind::SpikingBlock,
            PortSpec::any(),
            // per-timestep spike train over n_neurons
            PortSpec::f32(vec![
                DimSpec::Symbolic("timesteps".into()),
                DimSpec::Fixed(n_neurons),
            ]),
        );
        g.set_attr(activity, "role", "SpikeActivity");

        let project = g.add_stage(
            "adapt.project_activity",
            StageKind::Adaptation,
            PortSpec::f32(vec![
                DimSpec::Symbolic("timesteps".into()),
                DimSpec::Fixed(n_neurons),
            ]),
            PortSpec::f32_exact(&[embed_dim]),
        );
        g.set_attr(project, "projection_mode", format!("{mode:?}"));
        g.set_attr(project, "role", "projector::project_spike_activity");

        let router = g.add_stage(
            "moe.router",
            StageKind::MoeRouter,
            PortSpec::f32_exact(&[embed_dim]),
            PortSpec::f32(vec![DimSpec::Symbolic("num_experts".into())]),
        );
        g.set_attr(router, "role", "expert_router.route");

        let readout = g.add_stage(
            "readout.experts",
            StageKind::Readout,
            PortSpec::f32(vec![DimSpec::Symbolic("num_experts".into())]),
            PortSpec::any(),
        );

        g.connect(activity, project);
        g.connect(project, router);
        g.connect(router, readout);
        g.compile()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_graph() -> StageGraph {
        let mut g = StageGraph::new();
        let a = g.add_stage(
            "embed",
            StageKind::Embedding,
            PortSpec::any(),
            PortSpec::f32(vec![DimSpec::Fixed(8)]),
        );
        let b = g.add_stage(
            "snn",
            StageKind::SpikingBlock,
            PortSpec::f32(vec![DimSpec::Fixed(8)]),
            PortSpec::any(),
        );
        g.connect(a, b);
        g
    }

    #[test]
    fn forward_plan_order_is_deterministic() {
        let mut g = linear_graph();
        let c = g.add_stage(
            "readout",
            StageKind::Readout,
            PortSpec::any(),
            PortSpec::any(),
        );
        g.connect(StageId(1), c);
        let plan = g.compile().unwrap();
        assert_eq!(
            plan.execution_order(),
            &[StageId(0), StageId(1), StageId(2)]
        );
    }

    #[test]
    fn parallel_branches_order_by_stage_id() {
        let mut g = StageGraph::new();
        let src = g.add_stage(
            "src",
            StageKind::Embedding,
            PortSpec::any(),
            PortSpec::any(),
        );
        // Add b2 first so it gets the lower ID; both branch off src.
        let b1 = g.add_stage("b1", StageKind::Attention, PortSpec::any(), PortSpec::any());
        let b2 = g.add_stage("b2", StageKind::DenseMlp, PortSpec::any(), PortSpec::any());
        let sink = g.add_stage("sink", StageKind::Readout, PortSpec::any(), PortSpec::any());
        g.connect(src, b2);
        g.connect(src, b1);
        g.connect(b1, sink);
        g.connect(b2, sink);
        let plan = g.compile().unwrap();
        assert_eq!(
            plan.execution_order(),
            &[src, b1, b2, sink],
            "ready set must pop lowest StageId first"
        );
    }

    #[test]
    fn compat_plan_represents_ann_to_snn() {
        let plan = HybridExecutionPlan::from_hybrid_config(&HybridConfig::tiny()).unwrap();
        assert_eq!(plan.stages().len(), 3);
        assert_eq!(
            plan.execution_order(),
            &[StageId(0), StageId(1), StageId(2)]
        );
        assert_eq!(plan.resolved_domain(StageId(0)), Some(ExecutionDomain::Ann));
        assert_eq!(plan.resolved_domain(StageId(2)), Some(ExecutionDomain::Snn));
        assert!(plan.describe().contains("adapt.project_stimuli"));
    }

    #[test]
    fn compat_plan_represents_reverse_flow() {
        let plan = HybridExecutionPlan::from_reverse_path(ProjectionMode::RateSum, 32, 64).unwrap();
        assert_eq!(plan.stages().len(), 4);
        let project = plan.stage_by_name("adapt.project_activity").unwrap();
        assert_eq!(project.attrs.get("projection_mode").unwrap(), "RateSum");
        assert_eq!(plan.resolved_domain(StageId(2)), Some(ExecutionDomain::Moe));
    }

    #[test]
    fn rejects_empty_graph() {
        assert_eq!(StageGraph::new().compile().unwrap_err(), PlanError::Empty);
    }

    #[test]
    fn rejects_duplicate_names() {
        let mut g = linear_graph();
        g.add_stage(
            "embed",
            StageKind::Attention,
            PortSpec::any(),
            PortSpec::any(),
        );
        assert_eq!(
            g.compile().unwrap_err(),
            PlanError::DuplicateStageName {
                name: "embed".into()
            }
        );
    }

    #[test]
    fn rejects_disconnected_stage() {
        let mut g = linear_graph();
        g.add_stage(
            "orphan",
            StageKind::DenseMlp,
            PortSpec::any(),
            PortSpec::any(),
        );
        match g.compile().unwrap_err() {
            PlanError::Disconnected { stage, name } => {
                assert_eq!(stage, StageId(2));
                assert_eq!(name, "orphan");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn rejects_forward_cycle() {
        let mut g = StageGraph::new();
        let a = g.add_stage("a", StageKind::DenseMlp, PortSpec::any(), PortSpec::any());
        let b = g.add_stage("b", StageKind::DenseMlp, PortSpec::any(), PortSpec::any());
        g.connect(a, b);
        g.connect(b, a);
        assert_eq!(
            g.compile().unwrap_err(),
            PlanError::Cyclic { stages: vec![a, b] }
        );
    }

    #[test]
    fn feedback_requires_opt_in() {
        let mut g = StageGraph::new();
        let a = g.add_stage(
            "ann",
            StageKind::Transformer,
            PortSpec::any(),
            PortSpec::any(),
        );
        let b = g.add_stage(
            "snn",
            StageKind::SpikingBlock,
            PortSpec::any(),
            PortSpec::any(),
        );
        g.connect(a, b);
        g.connect_feedback(b, a);
        match g.compile().unwrap_err() {
            PlanError::FeedbackNotPermitted { from, to } => {
                assert_eq!((from, to), (b, a));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn feedback_edge_allowed_when_opted_in() {
        let mut g = StageGraph::new();
        let a = g.add_stage(
            "ann",
            StageKind::Transformer,
            PortSpec::any(),
            PortSpec::any(),
        );
        let b = g.add_stage(
            "snn",
            StageKind::SpikingBlock,
            PortSpec::any(),
            PortSpec::any(),
        );
        g.allow_feedback(true);
        g.connect(a, b);
        g.connect_feedback(b, a);
        let plan = g.compile().unwrap();
        // Feedback does not break the forward DAG.
        assert_eq!(plan.execution_order(), &[a, b]);
        assert_eq!(plan.feedback_inputs(a).len(), 1);
        assert!(plan.allows_feedback());
    }

    #[test]
    fn rejects_incompatible_contract() {
        let mut g = StageGraph::new();
        let a = g.add_stage(
            "a",
            StageKind::DenseMlp,
            PortSpec::any(),
            PortSpec::f32_exact(&[16]),
        );
        let b = g.add_stage(
            "b",
            StageKind::SpikingBlock,
            PortSpec::f32_exact(&[8]),
            PortSpec::any(),
        );
        g.connect(a, b);
        match g.compile().unwrap_err() {
            PlanError::IncompatibleContract { from, to, reason } => {
                assert_eq!((from, to), (a, b));
                assert!(reason.contains("16"), "{reason}");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn symbolic_dims_unify_across_edges() {
        let mut g = StageGraph::new();
        let a = g.add_stage(
            "a",
            StageKind::Embedding,
            PortSpec::any(),
            PortSpec::f32(vec![DimSpec::Symbolic("w".into())]),
        );
        let b = g.add_stage(
            "b",
            StageKind::DenseMlp,
            PortSpec::f32(vec![DimSpec::Fixed(4)]),
            PortSpec::f32(vec![DimSpec::Symbolic("w".into())]),
        );
        let c = g.add_stage(
            "c",
            StageKind::Readout,
            PortSpec::f32(vec![DimSpec::Fixed(4)]),
            PortSpec::any(),
        );
        g.connect(a, b);
        g.connect(b, c);
        g.compile().unwrap();

        // Conflicting binding of the same symbol fails.
        let mut g2 = StageGraph::new();
        let a2 = g2.add_stage(
            "a",
            StageKind::Embedding,
            PortSpec::any(),
            PortSpec::f32(vec![DimSpec::Fixed(4)]),
        );
        let b2 = g2.add_stage(
            "b",
            StageKind::DenseMlp,
            PortSpec::f32(vec![DimSpec::Symbolic("w".into())]),
            PortSpec::f32(vec![DimSpec::Fixed(9)]),
        );
        let c2 = g2.add_stage(
            "c",
            StageKind::Readout,
            PortSpec::f32(vec![DimSpec::Symbolic("w".into())]),
            PortSpec::any(),
        );
        g2.connect(a2, b2);
        g2.connect(b2, c2);
        assert!(matches!(
            g2.compile().unwrap_err(),
            PlanError::IncompatibleContract { .. }
        ));
    }

    #[test]
    fn rejects_invalid_domain_assignment() {
        let mut g = linear_graph();
        assert!(g.set_domain(StageId(1), ExecutionDomain::Ann));
        match g.compile().unwrap_err() {
            PlanError::InvalidDomain { stage, domain, .. } => {
                assert_eq!(stage, StageId(1));
                assert_eq!(domain, ExecutionDomain::Ann);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn domain_hook_allows_permitted_override() {
        let mut g = linear_graph();
        // Adaptation may run in Adapter or Ann; an explicit Ann pin is legal.
        let adapt = g.add_stage(
            "adapt",
            StageKind::Adaptation,
            PortSpec::any(),
            PortSpec::any(),
        );
        g.connect(StageId(1), adapt);
        assert!(g.set_domain(adapt, ExecutionDomain::Ann));
        let plan = g.compile().unwrap();
        assert_eq!(plan.resolved_domain(adapt), Some(ExecutionDomain::Ann));
    }

    #[test]
    fn set_domain_rejects_unknown_id() {
        let mut g = linear_graph();
        assert!(!g.set_domain(StageId(99), ExecutionDomain::Snn));
    }

    #[test]
    fn rejects_zero_fixed_extent() {
        let mut g = StageGraph::new();
        g.add_stage(
            "bad",
            StageKind::DenseMlp,
            PortSpec::any(),
            PortSpec::f32(vec![DimSpec::Fixed(0)]),
        );
        assert!(matches!(
            g.compile().unwrap_err(),
            PlanError::InvalidContract { .. }
        ));
    }

    #[test]
    fn json_round_trip_is_deterministic() {
        let plan = HybridExecutionPlan::from_hybrid_config(&HybridConfig::tiny()).unwrap();
        let j1 = plan.to_json().unwrap();
        let back = HybridExecutionPlan::from_json(&j1).unwrap();
        assert_eq!(back, plan);
        let j2 = back.to_json().unwrap();
        assert_eq!(j1, j2, "serialization must be deterministic");
    }

    #[test]
    fn from_json_rejects_inconsistent_order() {
        let mut plan = HybridExecutionPlan::from_hybrid_config(&HybridConfig::tiny()).unwrap();
        plan.order.pop();
        let json = serde_json::to_string(&plan).unwrap();
        assert!(matches!(
            HybridExecutionPlan::from_json(&json).unwrap_err(),
            PlanError::Inconsistent(_)
        ));
    }

    #[test]
    fn from_hybrid_config_rejects_zero_dims() {
        let mut cfg = HybridConfig::tiny();
        cfg.snn_input_channels = 0;
        assert!(matches!(
            HybridExecutionPlan::from_hybrid_config(&cfg).unwrap_err(),
            PlanError::InvalidContract { .. }
        ));
    }

    #[test]
    fn from_json_rejects_non_topological_order() {
        let mut plan = HybridExecutionPlan::from_hybrid_config(&HybridConfig::tiny()).unwrap();
        plan.order.reverse();
        let json = serde_json::to_string(&plan).unwrap();
        assert!(matches!(
            HybridExecutionPlan::from_json(&json).unwrap_err(),
            PlanError::Inconsistent(_)
        ));
    }

    #[test]
    fn from_json_rejects_duplicate_order_ids() {
        let mut plan = HybridExecutionPlan::from_hybrid_config(&HybridConfig::tiny()).unwrap();
        // Duplicate s0, drop s2 — same length, same set membership, but not a
        // valid execution sequence.
        plan.order[2] = StageId(0);
        let json = serde_json::to_string(&plan).unwrap();
        assert!(matches!(
            HybridExecutionPlan::from_json(&json).unwrap_err(),
            PlanError::Inconsistent(_)
        ));
    }

    #[test]
    fn describe_is_deterministic() {
        let plan = HybridExecutionPlan::from_reverse_path(ProjectionMode::RateSum, 8, 16).unwrap();
        let a = plan.describe();
        let plan2 = HybridExecutionPlan::from_reverse_path(ProjectionMode::RateSum, 8, 16).unwrap();
        assert_eq!(a, plan2.describe());
        assert!(a.contains("edge Forward s0 -> s1"));
    }
}
