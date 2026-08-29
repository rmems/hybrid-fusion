// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration tests for the dry-run planning surface (`HybridStagePlanner`).
//!
//! File-free by contract: the planner sees stage **names** only — no manifests,
//! no weights, no machine-local paths, no network. Imports come from the crate
//! root so the public re-exports stay honest.

use hybrid_fusion::{
    HybridError, HybridStagePlanner, OperationKind, PlannedOperation, PrecisionTier, Result,
};

// ---------------------------------------------------------------------------
// Out-of-crate planner (always available)
// ---------------------------------------------------------------------------

/// One tier for everything, always a conversion.
///
/// Deliberately unlike the in-crate unit mock, so the trait is shown to be
/// implementable by a stranger without borrowing its rules.
struct FixedTierPlanner {
    tier: PrecisionTier,
}

impl HybridStagePlanner for FixedTierPlanner {
    fn default_tier(&self) -> PrecisionTier {
        self.tier
    }

    fn plan(&self, stages: &[&str]) -> Result<Vec<PlannedOperation>> {
        stages
            .iter()
            .map(|&stage| {
                if stage.is_empty() {
                    return Err(HybridError::InvalidConfig(
                        "FixedTierPlanner::plan: stage name must not be empty".into(),
                    ));
                }
                Ok(PlannedOperation {
                    stage: stage.to_string(),
                    tier: self.tier,
                    kind: OperationKind::Convert,
                })
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Contract tests
// ---------------------------------------------------------------------------

#[test]
fn stage_planner_plans_every_stage_in_order() {
    let planner = FixedTierPlanner {
        tier: PrecisionTier::TernarySnn,
    };
    let stages = ["blk.0.attn_q", "blk.0.moe_gate", "blk.0.expert.0"];
    let plan = planner.plan(&stages).unwrap();
    assert_eq!(plan.len(), stages.len());
    for (op, name) in plan.iter().zip(stages) {
        assert_eq!(op.stage, name);
        assert_eq!(op.tier, planner.default_tier());
        assert_eq!(op.kind, OperationKind::Convert);
    }
}

#[test]
fn stage_planner_empty_pipeline_is_ok() {
    let planner = FixedTierPlanner {
        tier: PrecisionTier::Preserve,
    };
    assert!(planner.plan(&[]).unwrap().is_empty());
}

#[test]
fn stage_planner_rejects_empty_stage_name() {
    let planner = FixedTierPlanner {
        tier: PrecisionTier::Fp16,
    };
    match planner.plan(&["ok", ""]).unwrap_err() {
        HybridError::InvalidConfig(msg) => assert!(msg.contains("stage name"), "{msg}"),
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

/// A plan travels to a quantizing backend as JSON; the tier names on the wire
/// must be the manifest names, verified from outside the crate.
#[test]
fn stage_planner_plan_serializes_with_manifest_tier_names() {
    let planner = FixedTierPlanner {
        tier: PrecisionTier::TernarySnn,
    };
    let plan = planner.plan(&["blk.0.expert.0"]).unwrap();
    let json = serde_json::to_string(&plan).unwrap();
    assert!(json.contains("\"ternary_snn\""), "{json}");
    assert!(!json.contains("TernarySnn"), "{json}");
}
