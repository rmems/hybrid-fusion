// SPDX-License-Identifier: MIT OR Apache-2.0

//! Format-neutral topology planning. No checkpoint file, loader, or layout type.

use hybrid_fusion::{
    AnnTopology, HybridConfig, HybridError, HybridExecutionPlan, HybridTopology, PlanError,
    PortSpec, ProjectionMode, SnnTopology, TopologySource,
};

fn plan_from_source(
    source: &impl TopologySource,
    input_channels: usize,
    output_population: usize,
) -> Result<HybridExecutionPlan, PlanError> {
    let ann = source.ann_topology()?;
    let topology = HybridTopology::new(ann, SnnTopology::new(input_channels, output_population)?)?;
    HybridExecutionPlan::from_topology(&topology)
}

struct FixedSource;

impl TopologySource for FixedSource {
    fn ann_topology(&self) -> Result<AnnTopology, PlanError> {
        AnnTopology::new(128, 64, Some("mock-arch".into()))
    }
}

struct MissingWidth;

impl TopologySource for MissingWidth {
    fn ann_topology(&self) -> Result<AnnTopology, PlanError> {
        Err(PlanError::InvalidTopology {
            field: "hidden_dim",
            reason: "hidden width is absent".into(),
        })
    }
}

fn sample_topology() -> HybridTopology {
    HybridTopology::new(
        AnnTopology::new(128, 7, Some("programmatic".into())).unwrap(),
        SnnTopology::new(3, 11).unwrap(),
    )
    .unwrap()
}

#[test]
fn programmatic_topology_matches_hybrid_config_plan() {
    let config = HybridConfig::tiny();
    let topology = HybridTopology::new(
        AnnTopology::new(config.transformer.dim, config.transformer.max_seq_len, None).unwrap(),
        SnnTopology::new(config.snn_input_channels, config.snn_lif_neurons).unwrap(),
    )
    .unwrap();
    let from_topology = HybridExecutionPlan::from_topology(&topology).unwrap();
    let from_config = HybridExecutionPlan::from_hybrid_config(&config).unwrap();
    assert_eq!(from_topology, from_config);
}

#[test]
fn snn_axes_stay_independent_and_architecture_is_descriptive() {
    let topology = sample_topology();
    let plan = HybridExecutionPlan::from_topology(&topology).unwrap();
    let ann = plan.stage_by_name("ann.transformer").unwrap();
    assert_eq!(
        ann.attrs.get("accepted_layouts").map(String::as_str),
        Some("[dim] | [seq, dim]")
    );
    assert_eq!(ann.attrs.get("last_axis").map(String::as_str), Some("128"));
    assert_eq!(ann.attrs.get("max_sequence").map(String::as_str), Some("7"));
    let adapt = plan.stage_by_name("adapt.project_stimuli").unwrap();
    assert_eq!(adapt.output, PortSpec::f32_exact(&[3]));
    assert_eq!(
        adapt.attrs.get("hidden_dim").map(String::as_str),
        Some("128")
    );
    let snn = plan.stage_by_name("snn.step").unwrap();
    assert_eq!(snn.input, PortSpec::f32_exact(&[3]));
    assert_eq!(snn.attrs.get("num_neurons").map(String::as_str), Some("11"));

    let mut relabeled = topology.clone();
    relabeled.ann.architecture = None;
    assert_eq!(
        HybridExecutionPlan::from_topology(&relabeled).unwrap(),
        plan
    );
}

#[test]
fn zero_dimensions_name_the_failing_field() {
    let cases = [
        (AnnTopology::new(0, 4, None).unwrap_err(), "hidden_dim"),
        (AnnTopology::new(4, 0, None).unwrap_err(), "max_seq_len"),
        (SnnTopology::new(0, 4).unwrap_err(), "input_channels"),
        (SnnTopology::new(4, 0).unwrap_err(), "output_population"),
    ];
    for (err, field) in cases {
        assert!(
            matches!(err, PlanError::InvalidTopology { field: got, .. } if got == field),
            "expected field {field}, got {err:?}"
        );
    }

    let ann = AnnTopology::new(4, 4, None).unwrap();
    let snn = SnnTopology::new(4, 4).unwrap();
    let mut topology = HybridTopology::new(ann, snn).unwrap();
    topology.snn.output_population = 0;
    assert!(matches!(
        HybridExecutionPlan::from_topology(&topology).unwrap_err(),
        PlanError::InvalidTopology {
            field: "output_population",
            ..
        }
    ));
}

#[test]
fn deserialization_rejects_zero_dimensions() {
    let ann_rejected = [
        (r#"{"hidden_dim":0,"max_seq_len":4}"#, "hidden_dim"),
        (
            r#"{"hidden_dim":4,"max_seq_len":0,"architecture":"x"}"#,
            "max_seq_len",
        ),
    ];
    for (json, field) in ann_rejected {
        let err = serde_json::from_str::<AnnTopology>(json).unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("invalid topology field") && text.contains(field),
            "{text}"
        );
    }

    let snn_rejected = [
        (
            r#"{"input_channels":0,"output_population":4}"#,
            "input_channels",
        ),
        (
            r#"{"input_channels":4,"output_population":0}"#,
            "output_population",
        ),
    ];
    for (json, field) in snn_rejected {
        let err = serde_json::from_str::<SnnTopology>(json).unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("invalid topology field") && text.contains(field),
            "{text}"
        );
    }

    let hybrid = r#"{
        "ann": {"hidden_dim": 0, "max_seq_len": 4, "architecture": null},
        "snn": {"input_channels": 2, "output_population": 3}
    }"#;
    let err = serde_json::from_str::<HybridTopology>(hybrid).unwrap_err();
    assert!(err.to_string().contains("hidden_dim"), "{err}");

    let ann = AnnTopology::new(8, 16, Some("llama".into())).unwrap();
    let back: AnnTopology = serde_json::from_str(&serde_json::to_string(&ann).unwrap()).unwrap();
    assert_eq!(back, ann);
}

#[test]
fn mock_source_compiles_without_a_checkpoint() {
    let plan = plan_from_source(&FixedSource, 64, 32).unwrap();
    let from_config = HybridExecutionPlan::from_hybrid_config(&HybridConfig::tiny()).unwrap();
    assert_eq!(plan, from_config);
}

#[test]
fn mock_source_missing_value_does_not_infer_a_plan() {
    let err = plan_from_source(&MissingWidth, 64, 32).unwrap_err();
    assert!(matches!(
        err,
        PlanError::InvalidTopology {
            field: "hidden_dim",
            ..
        }
    ));
    let carried: HybridError = err.into();
    assert!(matches!(
        carried,
        HybridError::ExecutionPlan(PlanError::InvalidTopology {
            field: "hidden_dim",
            ..
        })
    ));
}

#[test]
fn reverse_topology_matches_reverse_path() {
    let topology = sample_topology();
    let from_topology =
        HybridExecutionPlan::from_reverse_topology(&topology, ProjectionMode::RateSum, 128)
            .unwrap();
    let from_path =
        HybridExecutionPlan::from_reverse_path(ProjectionMode::RateSum, 11, 128).unwrap();
    assert_eq!(from_topology, from_path);
}

#[test]
fn reverse_embedding_width_is_independent_of_ann_hidden_width() {
    let topology = sample_topology();
    assert_eq!(topology.ann.hidden_dim, 128);
    let plan =
        HybridExecutionPlan::from_reverse_topology(&topology, ProjectionMode::RateSum, 48).unwrap();
    let project = plan.stage_by_name("adapt.project_activity").unwrap();
    assert_eq!(project.output, PortSpec::f32_exact(&[48]));
    let router = plan.stage_by_name("moe.router").unwrap();
    assert_eq!(router.input, PortSpec::f32_exact(&[48]));
    assert_eq!(
        plan.stage_by_name("snn.activity")
            .unwrap()
            .attrs
            .get("n_neurons")
            .map(String::as_str),
        Some("11")
    );
    let followed_ann_width =
        HybridExecutionPlan::from_reverse_path(ProjectionMode::RateSum, 11, 128).unwrap();
    assert_ne!(plan, followed_ann_width);
}

#[test]
fn topology_plan_json_round_trip() {
    let plan = HybridExecutionPlan::from_topology(&sample_topology()).unwrap();
    let json = plan.to_json().unwrap();
    let back = HybridExecutionPlan::from_json(&json).unwrap();
    assert_eq!(back, plan);
}
