// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validation of backend requirements against compiled stage port contracts.

use super::{BackendCapabilities, StageRequirement, incompatible_dtype};
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
    validate_requirement_dtype(req, stage)?;
    validate_last_dim(plan, req, "hidden_dim", req.hidden_dim, &stage.output.dims)
}

fn validate_transformer(
    plan: &HybridExecutionPlan,
    req: &StageRequirement,
    stage: &Stage,
) -> Result<(), PlanError> {
    validate_requirement_dtype(req, stage)?;
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
    validate_metadata_dim(req, stage, "hidden_dim", req.hidden_dim, &["last_axis"])?;
    validate_metadata_dim(
        req,
        stage,
        "max_sequence",
        req.max_sequence,
        &["max_sequence"],
    )
}

fn validate_spiking_block(
    plan: &HybridExecutionPlan,
    req: &StageRequirement,
    stage: &Stage,
) -> Result<(), PlanError> {
    validate_requirement_dtype(req, stage)?;
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
    validate_requirement_dtype(req, stage)?;
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
            validate_metadata_dim(req, stage, "hidden_dim", req.hidden_dim, &["hidden_dim"])?;
            validate_metadata_dim(req, stage, "num_neurons", req.num_neurons, &["num_neurons"])
        }
        StageKind::Readout => {
            validate_last_dim(plan, req, "hidden_dim", req.hidden_dim, &stage.input.dims)
        }
        StageKind::Embedding | StageKind::Transformer | StageKind::SpikingBlock => {
            unreachable!("handled by dedicated validator")
        }
    }
}

pub(super) fn validate_backend_dtypes(
    req: &StageRequirement,
    stage: &Stage,
    caps: &BackendCapabilities,
) -> Result<(), PlanError> {
    for dtype in numerical_port_dtypes(stage).into_iter().flatten() {
        if !caps.dtypes.contains(&dtype) {
            return Err(incompatible_dtype(req, dtype, caps));
        }
    }
    Ok(())
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

fn validate_requirement_dtype(req: &StageRequirement, stage: &Stage) -> Result<(), PlanError> {
    let Some(required) = req.dtype else {
        return Ok(());
    };
    let port_dtypes = numerical_port_dtypes(stage);
    if port_dtypes.iter().flatten().any(|dtype| *dtype == required) {
        return Ok(());
    }
    let contract = port_dtypes
        .into_iter()
        .flatten()
        .map(|dtype| format!("{dtype:?}"))
        .collect::<Vec<_>>()
        .join(" or ");
    if contract.is_empty() {
        Ok(())
    } else {
        contract_mismatch(req, "dtype", format!("{required:?}"), contract)
    }
}

fn numerical_port_dtypes(stage: &Stage) -> [Option<Dtype>; 2] {
    match stage.kind {
        // Token IDs and fired indices are index/control values rather than the
        // backend's numerical activation dtype.
        StageKind::Transformer | StageKind::Embedding => [stage.output.dtype, None],
        StageKind::SpikingBlock => [stage.input.dtype, None],
        StageKind::Attention
        | StageKind::DenseMlp
        | StageKind::MoeRouter
        | StageKind::MoeExperts
        | StageKind::Adaptation
        | StageKind::Readout => [stage.input.dtype, stage.output.dtype],
    }
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
