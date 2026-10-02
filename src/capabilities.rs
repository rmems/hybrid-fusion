// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend capability discovery and policy-driven negotiation (issue #41 /
//! RM-1805).
//!
//! Discovery is a pure report: [`crate::Transformer::capabilities`] and
//! [`crate::SpikingNetwork::capabilities`] must not run inference or mutate
//! backend state. Negotiation compares those reports to per-stage
//! [`StageRequirement`]s and either selects a backend or returns a structured
//! [`PlanError`]. Fallback is never implicit — a stage substitutes another
//! backend only when [`FallbackPolicy::AllowNamed`] names that backend.
//!
//! This module does not manage devices, select kernels, rank performance, or
//! know checkpoint formats. Device and acceleration fields are hints only.
//!
//! # RM-1421 reconciliation
//!
//! Linear RM-1421 ("backend substitution and capability negotiation audit")
//! never produced findings. Its only agent session died on a GitHub rate limit
//! on 2026-09-15 and was not relaunched (Linear comment, 2026-09-18). There is
//! therefore no audit finding list to port. The contract here is the one #41
//! asked for, aligned with the capability reports cortex-tensor already
//! publishes (`SnnCapabilities` / `AnnCapabilities` in `rmems/cortex-tensor`
//! `src/snn/mod.rs` and `src/stage.rs`):
//!
//! - `backend_name` is a static identity string, matching
//!   `"neuromod::SpikingNetwork"` on the SNN side.
//! - Supported dtypes and stage features are explicit sets, matching
//!   `supported_dtypes` / `supported_tags`.
//! - `stateful` matches `AnnCapabilities.stateful`.
//! - Reset, caller-controlled RNG, plasticity, neuromodulation, and frozen
//!   evaluation are optional features, matching the boolean flags on
//!   `SnnCapabilities` (`caller_rng`, `plasticity`, `neuromodulation`,
//!   `frozen_evaluation`) rather than a second parallel vocabulary.
//! - Device reporting is intentionally a hint, not a selector: cortex-tensor
//!   omits a device field pending separate work, and this crate does not
//!   manage devices either.
//!
//! Translating a live cortex `AnnExecutor` into this report is RM-1831, not
//! this issue.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::plan::{ExecutionDomain, HybridExecutionPlan, PlanError, StageId};
use crate::types::Dtype;

/// Stable identity of an offered backend. Compared by value, not by pointer.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BackendId(String);

impl BackendId {
    /// Identity from a caller-chosen name. Empty names are rejected at
    /// negotiation time, not here, so static construction stays infallible.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Borrowed name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for BackendId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Named optional capability a stage may require and a backend may advertise.
///
/// Names are matched exactly. The well-known set used by the in-tree adapters
/// is `step`, `reset`, `caller_rng`, `plasticity`, `neuromodulation`,
/// `frozen_evaluation`, and `reference-embedding`. Callers may use other names;
/// an unknown name is simply unsupported unless the backend lists it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RequiredFeature(String);

impl RequiredFeature {
    /// Feature name. Stored owned so reports can outlive the call.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Borrowed name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RequiredFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Where a backend is willing to run. A hint, not a device handle.
///
/// No variant here allocates a device, selects a kernel, or implies CUDA.
/// `Host` is the only placement this crate can describe; `Accelerated` records
/// that a backend *claims* an accelerator without this crate managing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DeviceHint {
    /// Ordinary host memory. The default for every in-tree backend.
    Host,
    /// Backend claims an accelerator. This crate does not select or own it.
    Accelerated,
}

/// Side-effect-free description of what one backend can execute.
///
/// Construct with [`Self::ann`] / [`Self::snn`] and the `with_*` builders.
/// Absence of a field means "not advertised": a requirement that needs it
/// fails, rather than assuming a default that happens to match.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendCapabilities {
    /// Static identity, e.g. `"neuromod::SpikingNetwork"`.
    pub backend_name: String,
    /// Domains this backend can be assigned to. Sorted, deduplicated.
    pub domains: Vec<ExecutionDomain>,
    /// Element types the backend accepts and produces.
    pub dtypes: BTreeSet<Dtype>,
    /// Hidden width, when the backend is an ANN stage with a fixed width.
    pub hidden_dim: Option<usize>,
    /// Stimulus / input-channel width, when the backend is an SNN stage.
    pub channels: Option<usize>,
    /// Longest sequence or temporal window the backend accepts.
    pub max_sequence: Option<usize>,
    /// Largest batch the backend accepts. `None` means batching is not advertised.
    pub max_batch: Option<usize>,
    /// Backend can accept a stream of chunks rather than one full sequence.
    pub streaming: bool,
    /// `execute` / `step` mutates backend state across calls.
    pub stateful: bool,
    /// Backend exposes a reset that restores pre-step dynamics.
    pub reset: bool,
    /// Placement hint. Never consulted as a compatibility requirement.
    pub device: DeviceHint,
    /// Optional features the backend advertises. Sorted by name, deduplicated.
    pub features: Vec<RequiredFeature>,
}

impl BackendCapabilities {
    fn bare(name: impl Into<String>, domain: ExecutionDomain) -> Self {
        Self {
            backend_name: name.into(),
            domains: vec![domain],
            dtypes: BTreeSet::new(),
            hidden_dim: None,
            channels: None,
            max_sequence: None,
            max_batch: None,
            streaming: false,
            stateful: false,
            reset: false,
            device: DeviceHint::Host,
            features: Vec::new(),
        }
    }

    /// ANN-domain report. Dtypes, width, and features are unset until added.
    pub fn ann(name: impl Into<String>) -> Self {
        Self::bare(name, ExecutionDomain::Ann)
    }

    /// SNN-domain report. Dtypes, width, and features are unset until added.
    pub fn snn(name: impl Into<String>) -> Self {
        Self::bare(name, ExecutionDomain::Snn)
    }

    /// Adapter-domain report (projection / pooling, not a numerical backend).
    pub fn adapter(name: impl Into<String>) -> Self {
        Self::bare(name, ExecutionDomain::Adapter)
    }

    /// MoE-domain report.
    pub fn moe(name: impl Into<String>) -> Self {
        Self::bare(name, ExecutionDomain::Moe)
    }

    /// Replace the advertised domain set.
    ///
    /// [`ExecutionDomain::Auto`] is dropped: it is an assignment placeholder,
    /// not a domain a backend can run in.
    pub fn with_domains(mut self, domains: impl IntoIterator<Item = ExecutionDomain>) -> Self {
        self.domains = unique_sorted(domains.into_iter().filter(|d| *d != ExecutionDomain::Auto));
        self
    }

    /// Replace the advertised dtype set.
    pub fn with_dtypes(mut self, dtypes: impl IntoIterator<Item = Dtype>) -> Self {
        self.dtypes = dtypes.into_iter().collect();
        self
    }

    /// Advertise a fixed hidden width.
    pub fn with_hidden_dim(mut self, dim: usize) -> Self {
        self.hidden_dim = Some(dim);
        self
    }

    /// Advertise a fixed stimulus width.
    pub fn with_channels(mut self, channels: usize) -> Self {
        self.channels = Some(channels);
        self
    }

    /// Advertise the longest accepted sequence or temporal window.
    pub fn with_max_sequence(mut self, len: usize) -> Self {
        self.max_sequence = Some(len);
        self
    }

    /// Advertise batching up to `max_batch` (must be `> 0` to be useful).
    pub fn with_batching(mut self, enabled: bool) -> Self {
        self.max_batch = if enabled { Some(1) } else { None };
        self
    }

    /// Advertise a maximum batch greater than one.
    pub fn with_max_batch(mut self, max_batch: usize) -> Self {
        self.max_batch = Some(max_batch);
        self
    }

    /// Advertise streaming input.
    pub fn with_streaming(mut self, enabled: bool) -> Self {
        self.streaming = enabled;
        self
    }

    /// Advertise cross-call state.
    pub fn with_stateful(mut self, enabled: bool) -> Self {
        self.stateful = enabled;
        self
    }

    /// Advertise reset support.
    pub fn with_reset(mut self, enabled: bool) -> Self {
        self.reset = enabled;
        self
    }

    /// Record a placement hint. Not a compatibility input.
    pub fn with_device(mut self, device: DeviceHint) -> Self {
        self.device = device;
        self
    }

    /// Replace the advertised feature set.
    pub fn with_features(mut self, features: impl IntoIterator<Item = RequiredFeature>) -> Self {
        self.features = unique_sorted(features);
        self
    }

    /// Whether `domain` is in the advertised set. `Auto` is never advertised.
    pub fn supports_domain(&self, domain: ExecutionDomain) -> bool {
        self.domains.contains(&domain)
    }
}

fn unique_sorted<T: Ord>(items: impl IntoIterator<Item = T>) -> Vec<T> {
    let mut items: Vec<T> = items.into_iter().collect();
    items.sort();
    items.dedup();
    items
}

/// What one stage needs from a backend before execution.
///
/// Fields left unset are not checked. A set field is a hard requirement:
/// the selected backend must advertise a compatible value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageRequirement {
    stage: StageId,
    domain: ExecutionDomain,
    dtype: Option<Dtype>,
    hidden_dim: Option<usize>,
    channels: Option<usize>,
    max_sequence: Option<usize>,
    max_batch: Option<usize>,
    streaming: bool,
    stateful: bool,
    reset: bool,
    features: Vec<RequiredFeature>,
}

impl StageRequirement {
    /// Requirement that the stage run in `domain`. Other constraints are unset.
    pub fn new(stage: StageId, domain: ExecutionDomain) -> Self {
        Self {
            stage,
            domain,
            dtype: None,
            hidden_dim: None,
            channels: None,
            max_sequence: None,
            max_batch: None,
            streaming: false,
            stateful: false,
            reset: false,
            features: Vec::new(),
        }
    }

    /// Stage this requirement names.
    pub fn stage(&self) -> StageId {
        self.stage
    }

    /// Required domain.
    pub fn domain(&self) -> ExecutionDomain {
        self.domain
    }

    /// Require this element type.
    pub fn with_dtype(mut self, dtype: Dtype) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Require this hidden width (exact).
    pub fn with_hidden_dim(mut self, dim: usize) -> Self {
        self.hidden_dim = Some(dim);
        self
    }

    /// Require this stimulus width (exact).
    pub fn with_channels(mut self, channels: usize) -> Self {
        self.channels = Some(channels);
        self
    }

    /// Require a sequence window of at least `len`.
    pub fn with_max_sequence(mut self, len: usize) -> Self {
        self.max_sequence = Some(len);
        self
    }

    /// Require a batch of at least `batch`.
    pub fn with_batch(mut self, batch: usize) -> Self {
        self.max_batch = Some(batch);
        self
    }

    /// Require streaming input.
    pub fn requires_streaming(mut self) -> Self {
        self.streaming = true;
        self
    }

    /// Require cross-call state.
    pub fn requires_stateful(mut self) -> Self {
        self.stateful = true;
        self
    }

    /// Require reset support.
    pub fn requires_reset(mut self) -> Self {
        self.reset = true;
        self
    }

    /// Require one advertised feature.
    pub fn with_feature(mut self, feature: RequiredFeature) -> Self {
        if !self.features.iter().any(|f| f == &feature) {
            self.features.push(feature);
        }
        self
    }
}

/// Whether a rejected preferred backend may be replaced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FallbackPolicy {
    /// Never substitute. A preferred backend that cannot meet the requirement
    /// is an error even if another offered backend could.
    #[default]
    Forbid,
    /// Substitute only the backend named by [`CapabilityNegotiation::fallback_to`]
    /// for that stage. Unnamed backends are not considered.
    AllowNamed,
}

/// Per-stage result of a successful negotiation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NegotiationOutcome {
    /// The preferred (or only domain-matching) backend was selected.
    Selected(BackendId),
    /// The preferred backend was rejected and the named fallback was selected.
    Fallback {
        /// Backend that failed the requirement.
        rejected: BackendId,
        /// Named fallback that satisfied it.
        selected: BackendId,
    },
}

impl NegotiationOutcome {
    /// Backend that will execute the stage.
    pub fn selected(&self) -> &BackendId {
        match self {
            Self::Selected(id) | Self::Fallback { selected: id, .. } => id,
        }
    }
}

/// Successful negotiation: one outcome per required stage, in requirement order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NegotiationReport {
    outcomes: Vec<(StageId, NegotiationOutcome)>,
}

impl NegotiationReport {
    /// Outcome for `stage`, if that stage was in the requirement list.
    pub fn selection(&self, stage: StageId) -> Option<&NegotiationOutcome> {
        self.outcomes
            .iter()
            .find(|(id, _)| *id == stage)
            .map(|(_, outcome)| outcome)
    }

    /// Stages that used a named fallback, in requirement order.
    pub fn fallbacks(&self) -> Vec<StageId> {
        self.outcomes
            .iter()
            .filter(|(_, outcome)| matches!(outcome, NegotiationOutcome::Fallback { .. }))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Every outcome, in requirement order.
    pub fn outcomes(&self) -> &[(StageId, NegotiationOutcome)] {
        &self.outcomes
    }
}

#[derive(Debug, Clone)]
struct Offer {
    id: BackendId,
    caps: BackendCapabilities,
}

#[derive(Debug, Clone)]
struct Preference {
    stage: StageId,
    preferred: BackendId,
    fallback: Option<BackendId>,
}

/// Offered backends plus a fallback policy, applied to one plan.
///
/// Offering a backend does not run it. [`Self::negotiate`] only reads the
/// supplied [`BackendCapabilities`] values and the plan's stage IDs.
#[derive(Debug, Clone)]
pub struct CapabilityNegotiation {
    policy: FallbackPolicy,
    offers: Vec<Offer>,
    preferences: Vec<Preference>,
}

impl CapabilityNegotiation {
    /// Empty offer set under `policy`.
    pub fn new(policy: FallbackPolicy) -> Self {
        Self {
            policy,
            offers: Vec::new(),
            preferences: Vec::new(),
        }
    }

    /// Offer a backend report. A repeated id replaces the earlier report.
    pub fn offer(mut self, id: BackendId, caps: BackendCapabilities) -> Self {
        if let Some(existing) = self.offers.iter_mut().find(|o| o.id == id) {
            existing.caps = caps;
        } else {
            self.offers.push(Offer { id, caps });
        }
        self
    }

    /// Name the backend to try first for `stage`.
    pub fn prefer(mut self, stage: StageId, id: BackendId) -> Self {
        if let Some(pref) = self.preferences.iter_mut().find(|p| p.stage == stage) {
            pref.preferred = id;
        } else {
            self.preferences.push(Preference {
                stage,
                preferred: id,
                fallback: None,
            });
        }
        self
    }

    /// Name the only backend that may replace a rejected preferred backend.
    ///
    /// Ignored when the policy is [`FallbackPolicy::Forbid`]. Calling this
    /// without [`prefer`](Self::prefer) does not select `id`: a fallback is a
    /// substitute for a rejected preference, not a preference itself.
    pub fn fallback_to(mut self, stage: StageId, id: BackendId) -> Self {
        if let Some(pref) = self.preferences.iter_mut().find(|p| p.stage == stage) {
            pref.fallback = Some(id);
        } else {
            self.preferences.push(Preference {
                stage,
                preferred: BackendId::new(""),
                fallback: Some(id),
            });
        }
        self
    }

    /// Validate `requirements` against the offered reports and `plan`.
    ///
    /// `plan` is used to reject a requirement that names a stage the plan does
    /// not contain. It is not executed.
    ///
    /// # Errors
    ///
    /// - [`PlanError::UnknownStage`] if a requirement names a missing stage.
    /// - [`PlanError::UnsupportedBackend`] if no offered backend advertises the
    ///   required domain (or the preferred one does not, and fallback is forbidden
    ///   or unnamed).
    /// - [`PlanError::IncompatibleCapability`] for a dtype, width, sequence,
    ///   batch, streaming, stateful, or reset mismatch.
    /// - [`PlanError::UnsupportedFeature`] when a required feature is absent.
    /// - [`PlanError::SemanticMismatch`] when a named fallback fits the shape
    ///   but not the required feature set (substitution would change semantics).
    pub fn negotiate(
        &self,
        plan: &HybridExecutionPlan,
        requirements: &[StageRequirement],
    ) -> Result<NegotiationReport, PlanError> {
        let mut outcomes = Vec::with_capacity(requirements.len());
        for req in requirements {
            if plan.stage(req.stage).is_none() {
                return Err(PlanError::UnknownStage { stage: req.stage });
            }
            outcomes.push((req.stage, self.select(req)?));
        }
        Ok(NegotiationReport { outcomes })
    }

    fn select(&self, req: &StageRequirement) -> Result<NegotiationOutcome, PlanError> {
        let pref = self.preferences.iter().find(|p| p.stage == req.stage);
        if let Some(pref) = pref {
            return self.select_preferred(req, pref);
        }
        // A single offer is the implicit preference. Multiple offers without
        // prefer() stay ambiguous — picking the first would hide substitution.
        if self.offers.len() == 1 {
            let only = &self.offers[0];
            return match compatible(req, &only.caps) {
                Ok(()) => Ok(NegotiationOutcome::Selected(only.id.clone())),
                Err(err) => Err(err),
            };
        }
        // No preference: the unique domain-matching compatible backend, if any.
        let mut matches = self
            .offers
            .iter()
            .filter(|o| compatible(req, &o.caps).is_ok());
        let first = matches.next();
        let second = matches.next();
        match (first, second) {
            (Some(offer), None) => Ok(NegotiationOutcome::Selected(offer.id.clone())),
            (None, _) => Err(PlanError::UnsupportedBackend {
                stage: req.stage,
                domain: req.domain,
                reason: format!("no offered backend supports domain {:?}", req.domain),
            }),
            (Some(_), Some(_)) => Err(PlanError::UnsupportedBackend {
                stage: req.stage,
                domain: req.domain,
                reason: "multiple compatible backends; name one with prefer()".into(),
            }),
        }
    }

    fn select_preferred(
        &self,
        req: &StageRequirement,
        pref: &Preference,
    ) -> Result<NegotiationOutcome, PlanError> {
        let Some(preferred) = self.offers.iter().find(|o| o.id == pref.preferred) else {
            return Err(PlanError::UnsupportedBackend {
                stage: req.stage,
                domain: req.domain,
                reason: format!("preferred backend '{}' was not offered", pref.preferred),
            });
        };
        if let Err(err) = compatible(req, &preferred.caps) {
            return self.fallback(req, pref, err);
        }
        Ok(NegotiationOutcome::Selected(preferred.id.clone()))
    }

    fn fallback(
        &self,
        req: &StageRequirement,
        pref: &Preference,
        preferred_err: PlanError,
    ) -> Result<NegotiationOutcome, PlanError> {
        if self.policy != FallbackPolicy::AllowNamed {
            return Err(preferred_err);
        }
        let Some(fallback_id) = &pref.fallback else {
            return Err(preferred_err);
        };
        if fallback_id == &pref.preferred {
            return Err(preferred_err);
        }
        let Some(fallback) = self.offers.iter().find(|o| &o.id == fallback_id) else {
            return Err(PlanError::UnsupportedBackend {
                stage: req.stage,
                domain: req.domain,
                reason: format!("named fallback '{fallback_id}' was not offered"),
            });
        };
        // A fallback that misses a required feature is a semantic mismatch, not
        // a silent shape-only substitution — even when widths and dtypes match.
        if let Err(err) = compatible(req, &fallback.caps) {
            if matches!(err, PlanError::UnsupportedFeature { .. }) && shape_ok(req, &fallback.caps)
            {
                return Err(PlanError::SemanticMismatch {
                    stage: req.stage,
                    rejected: pref.preferred.as_str().to_string(),
                    candidate: fallback.id.as_str().to_string(),
                    reason: err.to_string(),
                });
            }
            return Err(err);
        }
        Ok(NegotiationOutcome::Fallback {
            rejected: pref.preferred.clone(),
            selected: fallback.id.clone(),
        })
    }
}

fn shape_ok(req: &StageRequirement, caps: &BackendCapabilities) -> bool {
    extent_ok(req.hidden_dim, caps.hidden_dim, true)
        && extent_ok(req.channels, caps.channels, true)
        && dtype_ok(req, caps)
        && caps.supports_domain(req.domain)
}

fn dtype_ok(req: &StageRequirement, caps: &BackendCapabilities) -> bool {
    match req.dtype {
        Some(dtype) => caps.dtypes.contains(&dtype),
        None => true,
    }
}

/// `exact` widths must match; sequence/batch windows must be at least the requirement.
fn extent_ok(required: Option<usize>, advertised: Option<usize>, exact: bool) -> bool {
    match (required, advertised) {
        (None, _) => true,
        (Some(_), None) => false,
        (Some(need), Some(have)) if exact => need == have,
        (Some(need), Some(have)) => have >= need,
    }
}

fn compatible(req: &StageRequirement, caps: &BackendCapabilities) -> Result<(), PlanError> {
    if !caps.supports_domain(req.domain) {
        return Err(PlanError::UnsupportedBackend {
            stage: req.stage,
            domain: req.domain,
            reason: format!(
                "backend '{}' does not advertise domain {:?}",
                caps.backend_name, req.domain
            ),
        });
    }
    if let Some(dtype) = req.dtype
        && !caps.dtypes.contains(&dtype)
    {
        let advertised = caps
            .dtypes
            .iter()
            .map(|d| format!("{d:?}"))
            .collect::<Vec<_>>()
            .join(",");
        return Err(PlanError::IncompatibleCapability {
            stage: req.stage,
            field: "dtype",
            required: format!("{dtype:?}"),
            backend: if advertised.is_empty() {
                "none".into()
            } else {
                advertised
            },
        });
    }
    check_exact(req, "hidden_dim", req.hidden_dim, caps.hidden_dim)?;
    check_exact(req, "channels", req.channels, caps.channels)?;
    check_at_least(req, "max_sequence", req.max_sequence, caps.max_sequence)?;
    check_at_least(req, "max_batch", req.max_batch, caps.max_batch)?;
    if req.streaming && !caps.streaming {
        return Err(PlanError::IncompatibleCapability {
            stage: req.stage,
            field: "streaming",
            required: "true".into(),
            backend: "false".into(),
        });
    }
    if req.stateful && !caps.stateful {
        return Err(PlanError::IncompatibleCapability {
            stage: req.stage,
            field: "stateful",
            required: "true".into(),
            backend: "false".into(),
        });
    }
    if req.reset && !caps.reset {
        return Err(PlanError::IncompatibleCapability {
            stage: req.stage,
            field: "reset",
            required: "true".into(),
            backend: "false".into(),
        });
    }
    for feature in &req.features {
        if !caps.features.iter().any(|f| f == feature) {
            return Err(PlanError::UnsupportedFeature {
                stage: req.stage,
                feature: feature.clone(),
            });
        }
    }
    Ok(())
}

fn check_exact(
    req: &StageRequirement,
    field: &'static str,
    required: Option<usize>,
    advertised: Option<usize>,
) -> Result<(), PlanError> {
    match (required, advertised) {
        (None, _) => Ok(()),
        (Some(need), Some(have)) if need == have => Ok(()),
        (Some(need), other) => Err(PlanError::IncompatibleCapability {
            stage: req.stage,
            field,
            required: need.to_string(),
            backend: other
                .map(|v| v.to_string())
                .unwrap_or_else(|| "unset".into()),
        }),
    }
}

fn check_at_least(
    req: &StageRequirement,
    field: &'static str,
    required: Option<usize>,
    advertised: Option<usize>,
) -> Result<(), PlanError> {
    match (required, advertised) {
        (None, _) => Ok(()),
        (Some(need), Some(have)) if have >= need => Ok(()),
        (Some(need), other) => Err(PlanError::IncompatibleCapability {
            stage: req.stage,
            field,
            required: need.to_string(),
            backend: other
                .map(|v| v.to_string())
                .unwrap_or_else(|| "unset".into()),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::StageGraph;
    use crate::types::Dtype;

    fn one_stage() -> (HybridExecutionPlan, StageId) {
        let mut g = StageGraph::new();
        let id = g.add_stage(
            "tower",
            crate::plan::StageKind::Transformer,
            crate::plan::PortSpec::any(),
            crate::plan::PortSpec::any(),
        );
        (g.compile().unwrap(), id)
    }

    #[test]
    fn streaming_requirement_names_the_field() {
        let (plan, stage) = one_stage();
        let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
            .offer(
                BackendId::new("batch-only"),
                BackendCapabilities::ann("batch-only").with_dtypes([Dtype::F32]),
            )
            .negotiate(
                &plan,
                &[StageRequirement::new(stage, ExecutionDomain::Ann).requires_streaming()],
            )
            .unwrap_err();
        match err {
            PlanError::IncompatibleCapability { field, .. } => assert_eq!(field, "streaming"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn batch_shorter_than_required_is_rejected() {
        let (plan, stage) = one_stage();
        let err = CapabilityNegotiation::new(FallbackPolicy::Forbid)
            .offer(
                BackendId::new("b1"),
                BackendCapabilities::ann("b1")
                    .with_dtypes([Dtype::F32])
                    .with_max_batch(1),
            )
            .negotiate(
                &plan,
                &[StageRequirement::new(stage, ExecutionDomain::Ann).with_batch(4)],
            )
            .unwrap_err();
        match err {
            PlanError::IncompatibleCapability {
                field,
                required,
                backend,
                ..
            } => {
                assert_eq!(field, "max_batch");
                assert_eq!(required, "4");
                assert_eq!(backend, "1");
            }
            other => panic!("{other:?}"),
        }
    }
}
