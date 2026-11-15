# Stage-graph and backend-capability conformance matrix

Linear [RM-1813](https://linear.app/rpd-34/issue/RM-1813). One map from every
stage kind to its execution domain, the capability outcome negotiation must
produce, and the test that covers the row — so coverage gaps are visible
instead of implied.

`tests/conformance_matrix.rs` is the executable side: `expected_domains` is an
exhaustive match over `StageKind`, so a new variant fails to compile until a
row is added there and here, and `negotiation_outcome_matrix` runs every kind
through the select / wrong-domain / missing-feature outcomes.

Legend for the file a covering test lives in:

- `cm` — `tests/conformance_matrix.rs` (this matrix)
- `cn` — `tests/capability_negotiation.rs`
- `ep` — `tests/execution_plan.rs`
- `tp` — `tests/topology_plan.rs`
- `hn` — `tests/hybrid_network.rs`
- `np` — `tests/neuromod_parity.rs` (`neuromod backends`)
- `smoke` — `smoke/cortex-neuromod/tests/` (CI-gated, fetches cortex-tensor)
- `plan`, `caps`, `snn`, `nmod`, `sr` — the `#[cfg(test)]` modules of
  `src/plan.rs`, `src/capabilities.rs`, `src/backends/simple_snn.rs`,
  `src/neuromod_adapter.rs`, and `src/spiking_router.rs`

## 1. Stage kind → domain → outcome

| StageKind | Default domain | Also allowed | Structural coverage | Negotiation coverage |
|---|---|---|---|---|
| `Embedding` | `Ann` | — | cm `stage_domain_matrix_is_total` | cm `negotiation_outcome_matrix`; cn `embedding_requirement_uses_activation_not_token_id_dtype`, `token_width_is_transient_and_never_declared` |
| `Attention` | `Ann` | — | cm | cm; cn `every_stage_kind_validates_its_numerical_port_dtype` |
| `DenseMlp` | `Ann` | — | cm | cm; cn `every_stage_kind_validates_its_numerical_port_dtype`, `numerical_stage_width_does_not_match_an_unrelated_axis` |
| `Transformer` | `Ann` | — | cm; plan `compat_plan_represents_ann_to_snn`; ep `hybrid_network_execution_plan_matches_forward_flow`; tp `compatible_ann_dimensions_tie_forced_architecture` | cm; cn `canonical_plan_metadata_constrains_backend_dimensions`, `canonical_plan_checks_backend_dtype_on_numerical_ports`, `canonical_transformer_requires_its_configured_sequence_capacity`, `canonical_transformer_accepts_a_shorter_declared_window`, `requirement_must_match_fixed_transformer_output_port` |
| `MoeRouter` | `Moe` | `Ann` | cm; plan `compat_plan_represents_reverse_flow`; ep `reverse_path_execution_plan_without_new_host` | cm; cn `every_stage_kind_validates_its_numerical_port_dtype` |
| `MoeExperts` | `Moe` | `Ann` | cm | cm; cn `every_stage_kind_validates_its_numerical_port_dtype` |
| `Adaptation` | `Adapter` | `Ann` | cm; plan `compat_plan_domain_overrides_from_topology`, `adapter_width_is_checked_before_contract` | cm; cn `canonical_forward_projector_requires_transformer_hidden_width`, `canonical_reverse_projector_requires_activity_population`, `dtype_conversion_requires_backend_support_for_both_ports`, `resolved_symbolic_width_constrains_negotiation` |
| `SpikingBlock` | `Snn` | — | cm; plan `rejects_incompatible_contract` | cm; cn `requirement_must_match_fixed_snn_input_port`, `channel_width_mismatch_names_both_sides`, `output_population_is_checked_apart_from_stimulus_width`, `reverse_path_neuron_metadata_constrains_negotiation` |
| `Readout` | `Ann` | `Adapter` | cm; plan `domain_hook_allows_permitted_override` | cm; cn `every_stage_kind_validates_its_numerical_port_dtype` |

Kind-independent structural errors — `Empty`, `DuplicateStageName`,
`NonSequentialStageId`, `Disconnected`, `Cyclic`, `FeedbackNotPermitted`,
`InvalidContract` (e.g. `Fixed(0)`), `IncompatibleContract` (extent, rank,
dtype, symbolic rebind, undeclared/mismatched spike encoding/decoding),
`Inconsistent` (serde re-validation), `InvalidTopology`, `InvalidParameters`,
`UnknownStage` — are covered by `src/plan.rs` unit tests and `ep`; `cm` adds
`UnknownStage` for both a dangling edge endpoint and a negotiation
requirement, which were previously uncovered.

## 2. Negotiation outcome matrix

| Scenario | Outcome | Covering test |
|---|---|---|
| Offer matches domain and contract | `Selected(backend)` | cm `negotiation_outcome_matrix`; cn `compatible_backend_is_selected_without_fallback` |
| Offer advertises a different domain | `UnsupportedBackend { domain }` | cm `negotiation_outcome_matrix`; cn `unsupported_domain_is_a_structured_error_not_a_string_scan` |
| dtype mismatch | `IncompatibleCapability { field: "dtype" }` | cn `dtype_and_width_mismatch_are_distinct_structured_errors` |
| hidden-width / channel / population mismatch | `IncompatibleCapability { field }` naming both sides | cn `dtype_and_width_mismatch_are_distinct_structured_errors`, `channel_width_mismatch_names_both_sides`, `output_population_is_checked_apart_from_stimulus_width` |
| sequence window shorter than required | `IncompatibleCapability { field: "max_sequence" }` | cn `sequence_window_shorter_than_required_is_rejected`, `sole_domain_match_keeps_its_capability_error` |
| streaming / batch hints unsatisfiable | `IncompatibleCapability { field }` | caps `streaming_requirement_names_the_field`, `batch_shorter_than_required_is_rejected` |
| Required feature absent from report | `UnsupportedFeature` | cm `negotiation_outcome_matrix`; cn `missing_optional_feature_fails_explicitly`, `unknown_feature_name_matches_no_report_that_omits_it` |
| Requirement domain ≠ resolved domain | `InvalidParameters` (pre-offer) | cm `requirement_domain_must_equal_the_resolved_domain`; cn `requirement_domain_must_match_the_compiled_stage` |
| Requirement contradicts the port contract or stage metadata | `InvalidParameters` | cn `requirement_must_match_fixed_snn_input_port`, `requirement_must_match_fixed_transformer_output_port`, `canonical_*`, `reverse_path_neuron_metadata_constrains_negotiation`, `resolved_symbolic_width_constrains_negotiation` |
| Requirement names a stage not in the plan | `UnknownStage` | cm `requirement_for_a_missing_stage_is_unknown_stage` |
| Edge endpoint names no stage | `UnknownStage` | cm `edge_to_a_missing_stage_is_unknown_stage` |
| ≥2 compatible offers, no `prefer()` | `UnsupportedBackend` ("multiple compatible backends…") | cm `unpreferred_compatible_offers_are_rejected_as_ambiguous` |
| `prefer()` names an offer not offered, no fallback | `UnsupportedBackend` ("preferred backend … was not offered") | cm `absent_preferred_backend_fails_structured_without_a_named_fallback` |
| `Forbid` policy, compatible/named fallback exists | error — never substitutes | cn `forbid_policy_does_not_silently_substitute_a_compatible_fallback`, `forbid_policy_never_substitutes_a_named_fallback` |
| `AllowNamed` + named fallback offered | `Fallback { rejected, selected }` recorded | cn `allow_policy_selects_the_named_fallback_and_records_it`, `named_fallback_runs_when_the_preferred_backend_was_not_offered` |
| `AllowNamed` + named fallback absent | `UnsupportedBackend` — no fall-through | cn `allow_named_absent_fallback_does_not_fall_through` |
| Fallback matches shape but misses features | `SemanticMismatch` | cn `semantic_inequivalence_rejects_substitution_even_when_shapes_match`, `fallback_that_also_misses_the_sequence_window_is_not_semantic` |
| Empty `BackendId` (offer / prefer / fallback) | `InvalidParameters` | cn `empty_backend_identity_is_rejected_before_selection`, `fallback_without_prefer_does_not_select_an_empty_identity` |
| Duplicate requirements for one stage | `InvalidParameters` | cn `duplicate_stage_requirements_are_rejected`, `identical_duplicate_stage_requirements_are_rejected` |
| `fallback_to` without `prefer()` | `InvalidParameters` before any offer is consulted | cn `malformed_fallback_preference_is_rejected_before_requirement_evaluation`, `fallback_without_prefer_is_invalid_before_candidate_evaluation`, `fallback_without_prefer_is_rejected_even_for_an_unrequired_stage` |
| Stage with no requirement | not in the report | cm `selection_returns_none_for_unrequired_stages` |

## 3. Dry-run plan inspection

`HybridExecutionPlan::describe()` is the rendered, reviewable form of a
compiled plan. `cm` goldens the forward (`from_hybrid_config`), reverse
(`from_reverse_path`), and feedback (`allow_feedback`) renders line-for-line,
and `plan_rendering_is_stable_across_rebuild_and_serde_round_trip` asserts
byte-identical `describe()`/`to_json()` across rebuilds and a `from_json`
round-trip. Ordering and serde determinism are additionally locked by plan
`forward_plan_order_is_deterministic`, `parallel_branches_order_by_stage_id`,
`json_round_trip_is_deterministic`, `describe_is_deterministic`, and the
`from_json_rejects_*` re-validation tests.

## 4. ANN↔SNN step contract (issue #61 / PR #65)

| Guarantee | Covering test |
|---|---|
| `step` consumes exactly `num_channels` stimuli; wrong width rejected | snn `rejects_stimuli_width_mismatch`; np `both_backends_reject_wrong_width` |
| Fired IDs in `0..num_neurons`; out-of-range → `SnnSpikeOutOfRange` | hn `fired_index_boundary_accepts_last_neuron_and_rejects_population_size`, `fired_bound_is_output_population_not_input_width`; sr `validate_fired` tests |
| Each ID at most once per tick → `SnnDuplicateSpike` | hn `duplicate_fired_index_is_rejected_without_advancing_host_counter`, `first_offending_fired_entry_determines_the_error` |
| `step_frozen` default fails closed; override advertises `frozen_evaluation` | cm `step_contract_defaults_fail_closed`, `frozen_evaluation_is_a_capability_gate_not_a_fallback`; sr `forward_frozen_fails_closed_without_frozen_capability` |
| `forward_frozen` gates on the advertised feature before the encoder advances | cm `frozen_evaluation_is_a_capability_gate_not_a_fallback`; sr `forward_frozen_uses_frozen_path_and_reset_clears_state` |
| `reset` default → `SnnUnsupported("reset")`; advertised ⇒ implemented | snn `trait_reset_is_callable_for_an_advertised_capability`, `trait_default_reset_errors_when_reset_is_not_advertised`; cm `step_contract_defaults_fail_closed` |
| Default SNN report is `step` + a one-step window only | cm `step_contract_defaults_fail_closed`; cn `default_snn_report_advertises_one_step_window` |
| `SpikeEncoding`/`SpikeDecoding` declared on `PortSpec` and matched across edges | plan `undeclared_producer_encoding_is_a_plan_error`, `matching_declarations_compile_and_mismatches_do_not`, `undeclared_producer_decoding_is_a_plan_error`, `port_spec_spike_fields_default_for_legacy_plans`; tp `snn_axes_stay_independent_and_architecture_is_descriptive`, `reverse_embedding_width_is_independent_of_ann_hidden_width` |
| Cross-crate parity with cortex-tensor's twin `src/snn` contract (ADR 0002) | smoke `snn_contract_parity.rs`, `cortex_neuromod_smoke.rs` |

## 5. In-tree report fidelity

| Report | Advertised | Deliberately not advertised | Evidence |
|---|---|---|---|
| `Transformer::capabilities` default (`SimpleTransformer`) | `Ann` domain, `F32`, `hidden_dim`, `max_sequence` | features, streaming, batching, stateful, reset | cm `simple_transformer_report_matches_the_trait_default` (`backends`) |
| `SpikingNetwork::capabilities` default | `Snn` domain, `F32`, `channels`, `num_neurons`, `max_sequence = 1`, `step` | everything else | cm `step_contract_defaults_fail_closed`; cn `default_snn_report_advertises_one_step_window` |
| `SimpleSnn` (mock) | default + `stateful`, `reset` | `caller_rng`, `plasticity`, `neuromodulation`, `frozen_evaluation` — the gain heuristic is not the neuromod feature set (module docs) | snn `capability_report_matches_step_and_reset`; cm `mock_snn_report_lists_only_implemented_behaviors` (`backends`) |
| `NeuromodSnn` (real) | the mock's set + `caller_rng`, `plasticity`, `neuromodulation`, `frozen_evaluation`; `backend_name = "neuromod::SpikingNetwork"` | — | nmod `capability_report_matches_step_reset_and_learning`, `advertised_caller_rng_drives_input_encoding`, `advertised_plasticity_and_neuromodulation_change_weights`; cm `mock_and_real_snn_reports_share_the_contract_fields`, `real_snn_only_features_gate_negotiation`, `advertised_real_features_are_implemented` (`neuromod backends`) |
| cortex `ReferenceExecutor` / `NeuromodNetwork` | cross-crate | — | smoke `cortex_neuromod_smoke.rs`, `snn_contract_parity.rs`; `AnnExecutor` → `BackendCapabilities` translation is RM-1831 |

Discovery is side-effect-free: cn `discovery_does_not_step_or_run_the_backend`,
nmod `capability_report_matches_step_reset_and_learning` (weights and the RNG
stream untouched), and cm `mock_and_real_snn_reports_share_the_contract_fields`.

## 6. RM-1421 reconciliation

The original RM-1421 audit produced no findings — its only agent session died
on a GitHub rate limit and was not relaunched (recorded in `src/capabilities.rs`
module docs). Its acceptance list is nonetheless covered:

| Guarantee (RM-1421 §3) | Covering test |
|---|---|
| Empty `BackendId` rejected | cn `empty_backend_identity_is_rejected_before_selection`, `fallback_without_prefer_does_not_select_an_empty_identity` |
| Duplicate stage requirements rejected | cn `duplicate_stage_requirements_are_rejected`, `identical_duplicate_stage_requirements_are_rejected` |
| Malformed fallback preferences rejected before selection | cn `malformed_fallback_preference_is_rejected_before_requirement_evaluation`, `fallback_without_prefer_is_invalid_before_candidate_evaluation`, `fallback_without_prefer_is_rejected_even_for_an_unrequired_stage` |
| Unknown `RequiredFeature` matches no report that omits it | cn `unknown_feature_name_matches_no_report_that_omits_it` |
| `FallbackPolicy::Forbid` never substitutes | cn `forbid_policy_does_not_silently_substitute_a_compatible_fallback`, `forbid_policy_never_substitutes_a_named_fallback` |
| `AllowNamed` naming an absent backend fails structured, not silent | cn `allow_named_absent_fallback_does_not_fall_through`; cm `absent_preferred_backend_fails_structured_without_a_named_fallback` |

Report fidelity (§1) is §5 above. The reconciliation-comment drift (§2) was
corrected by PR #57; the load-bearing claims — `backend_name`
`"neuromod::SpikingNetwork"`, the flag set, `stateful` — are asserted by nmod
`capability_report_matches_step_reset_and_learning` and cm
`mock_and_real_snn_reports_share_the_contract_fields`.

Cortex `AnnExecutor` fidelity and its capability translation belong to
RM-1831 / RM-1819 through the smoke harness — deliberately out of scope here.
