// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validation of backend requirements against compiled stage port contracts.

use super::StageRequirement;
use crate::plan::{DimSpec, HybridExecutionPlan, PlanError, Stage, StageKind};
use crate::types::Dtype;

pub(super) fn validate_stage_contract(
    plan: &HybridExecutionPlan,
    req: &StageRequirement,
    stage: &Stage,
) -> Result<(), PlanError> {
    match stage.kind {
        StageKind::Transformer => validate_transformer(plan, req, stage),
        StageKind::SpikingBlock => validate_spiking_block(plan, req, stage),
        StageKind::Embedding => validate_embedding(plan, req, stage),
        StageKind::Attention
        | StageKind::DenseMlp
        | StageKind::MoeRouter
        | StageKind::MoeExperts
        | StageKind::Adaptation
        | StageKind::Readout => validate_numerical_stage(plan, req, stage),
    }
}

fn validate_embedding(
    plan: &HybridExecutionPlan,
    req: &StageRequirement,
    stage: &Stage,
) -> Result<(), PlanError> {
    // Token IDs are an index/control port, not the embedding activation dtype.
    validate_port_dtype(req, "output dtype", stage.output.dtype)?;
    validate_last_dim(plan, req, "hidden_dim", req.hidden_dim, &stage.output.dims)
}

fn validate_transformer(
    plan: &HybridExecutionPlan,
    req: &StageRequirement,
    stage: &Stage,
) -> Result<(), PlanError> {
    validate_port_dtype(req, "output dtype", stage.output.dtype)?;
    if stage.output.dims.len() == 1 {
        validate_port_dim(
            plan,
            req,
            "hidden_dim",
            req.hidden_dim,
            &stage.output.dims[0],
        )?;
    } else if stage.output.dims.len() == 2 {
        validate_port_dim(
            plan,
            req,
            "max_sequence",
            req.max_sequence,
            &stage.output.dims[0],
        )?;
        validate_port_dim(
            plan,
            req,
            "hidden_dim",
            req.hidden_dim,
            &stage.output.dims[1],
        )?;
    }
    validate_metadata_dim(req, stage, "hidden_dim", req.hidden_dim, &["last_axis"])
}

fn validate_spiking_block(
    plan: &HybridExecutionPlan,
    req: &StageRequirement,
    stage: &Stage,
) -> Result<(), PlanError> {
    validate_port_dtype(req, "input dtype", stage.input.dtype)?;
    if stage.input.dims.len() == 1 {
        validate_port_dim(plan, req, "channels", req.channels, &stage.input.dims[0])?;
    }
    if stage.output.dims.len() == 1 {
        validate_port_dim(
            plan,
            req,
            "num_neurons",
            req.num_neurons,
            &stage.output.dims[0],
        )?;
    }
    validate_metadata_dim(
        req,
        stage,
        "num_neurons",
        req.num_neurons,
        &["num_neurons", "n_neurons"],
    )
}

fn validate_numerical_stage(
    plan: &HybridExecutionPlan,
    req: &StageRequirement,
    stage: &Stage,
) -> Result<(), PlanError> {
    validate_port_dtype(req, "input dtype", stage.input.dtype)?;
    validate_port_dtype(req, "output dtype", stage.output.dtype)?;
    match stage.kind {
        StageKind::Attention | StageKind::DenseMlp => {
            validate_last_dim(plan, req, "hidden_dim", req.hidden_dim, &stage.input.dims)?;
            validate_last_dim(plan, req, "hidden_dim", req.hidden_dim, &stage.output.dims)
        }
        StageKind::MoeRouter => {
            validate_last_dim(plan, req, "hidden_dim", req.hidden_dim, &stage.input.dims)
        }
        StageKind::MoeExperts => {
            validate_last_dim(plan, req, "hidden_dim", req.hidden_dim, &stage.output.dims)
        }
        StageKind::Adaptation => {
            validate_last_dim(plan, req, "num_neurons", req.num_neurons, &stage.input.dims)?;
            validate_last_dim(plan, req, "channels", req.channels, &stage.output.dims)?;
            if stage
                .attrs
                .get("role")
                .is_some_and(|role| role == "projector::project_spike_activity")
            {
                validate_last_dim(plan, req, "hidden_dim", req.hidden_dim, &stage.output.dims)?;
            }
            Ok(())
        }
        StageKind::Readout => {
            validate_last_dim(plan, req, "hidden_dim", req.hidden_dim, &stage.input.dims)
        }
        StageKind::Embedding | StageKind::Transformer | StageKind::SpikingBlock => {
            unreachable!("handled by dedicated validator")
        }
    }
}

fn validate_metadata_dim(
    req: &StageRequirement,
    stage: &Stage,
    field: &str,
    required: Option<usize>,
    attrs: &[&str],
) -> Result<(), PlanError> {
    let (Some(required), Some((attr, value))) = (
        required,
        attrs
            .iter()
            .find_map(|attr| stage.attrs.get(*attr).map(|value| (*attr, value))),
    ) else {
        return Ok(());
    };
    let contract = value.parse::<usize>().map_err(|_| {
        PlanError::InvalidParameters(format!(
            "stage {} has non-numeric {attr} metadata '{value}'",
            req.stage
        ))
    })?;
    if required != contract {
        return contract_mismatch(req, field, required.to_string(), contract.to_string());
    }
    Ok(())
}

fn validate_port_dtype(
    req: &StageRequirement,
    field: &str,
    port_dtype: Option<Dtype>,
) -> Result<(), PlanError> {
    if let (Some(required), Some(port)) = (req.dtype, port_dtype)
        && required != port
    {
        return contract_mismatch(req, field, format!("{required:?}"), format!("{port:?}"));
    }
    Ok(())
}

fn validate_port_dim(
    plan: &HybridExecutionPlan,
    req: &StageRequirement,
    field: &str,
    required: Option<usize>,
    port_dim: &DimSpec,
) -> Result<(), PlanError> {
    if let (Some(required), Some(port)) = (required, plan.resolved_dim(port_dim))
        && required != port
    {
        return contract_mismatch(req, field, required.to_string(), port.to_string());
    }
    Ok(())
}

fn validate_last_dim(
    plan: &HybridExecutionPlan,
    req: &StageRequirement,
    field: &str,
    required: Option<usize>,
    dims: &[DimSpec],
) -> Result<(), PlanError> {
    if let Some(dim) = dims.last() {
        validate_port_dim(plan, req, field, required, dim)?;
    }
    Ok(())
}

fn contract_mismatch(
    req: &StageRequirement,
    field: &str,
    required: String,
    port: String,
) -> Result<(), PlanError> {
    Err(PlanError::InvalidParameters(format!(
        "stage {} {field} port contract is {port}, requirement asks for {required}",
        req.stage
    )))
}
