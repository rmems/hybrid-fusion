# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] — v0.3.0 track

Package version is **0.3.0** (start of the LLM/SNN architecture-contracts series).
Trait/API landings for epic [#20](https://github.com/rmems/hybrid-fusion/issues/20)
will accumulate here until a tagged `v0.3.0` GitHub release.

### Added

- Optional off-by-default `neuromod` feature providing `NeuromodSnn`, a real SNN
  backend adapter over [`neuromod`](https://crates.io/crates/neuromod) 0.7's
  LIF/Izhikevich engine. It implements `SpikingNetwork` and supports seeded
  deterministic replay (`with_seed` plus a `step` routed through an owned RNG,
  backed by neuromod's `step_with_rng`), `reset()` (resets dynamics, does not
  reseed), and `reseed(seed)`. Enable with `cargo build --features neuromod`; the
  dependency and its transitive `rand` stay out of the default build. No neuromod
  type appears in any public signature: hybrid-fusion keeps its public
  `NeuroModulators { dopamine, cortisol, acetylcholine, tempo, aux_dopamine }` and
  the adapter converts explicitly (dopamine folds in `aux_dopamine` and clamps to
  `[0, 1]`; `cortisol` maps to neuromod `norepinephrine`; `acetylcholine` maps
  directly; `tempo` is intentionally unmapped as a step/timebase policy;
  `serotonin` stays at its `0.0` default) ([#44](https://github.com/rmems/hybrid-fusion/issues/44)).
- `HybridNetwork::try_new` validates transformer dim, max sequence length, and
  SNN input channels against backend-reported capabilities before the first
  forward call (Linear [RM-1357](https://linear.app/rpd-34/issue/RM-1357)).
- `HybridError::ConfigMismatch { field, configured, backend }` names the
  disagreeing construction field plus both values. Zero capacities use the
  same variant (`configured` and/or `backend` is `0`).
- Structured forward-path contract errors: `HiddenStateRank`, `HiddenStateSeqLen`,
  `HiddenStateDim`, `HiddenStateDataLen`, `NonFinite { stage, index }` (with
  `ForwardValueStage`), and `ZeroSnnChannels` ([#39](https://github.com/rmems/hybrid-fusion/pull/39)).
- `docs/extraction-map.md` — maps extractable LLM/SNN/MoE architecture from corinth-canal and grok-ozempic into hybrid-fusion traits vs sibling crates (#21).
- Dual checkpoint contract: `SafetensorsLoader` / `SafetensorsLayout` / `TensorManifestEntry`, plus `Dtype` (full Safetensors header vocabulary), `TensorRole` (name-heuristic classifier), and `HybridError::SafetensorsParse` ([#27](https://github.com/rmems/hybrid-fusion/issues/27)).
- Reverse-path contracts: `SpikeActivity`, `ExpertRouter`, `ExpertRouteOutput`; MoE fields on `HybridOutput`; `StubExpertRouter` under `backends` (#22).
- `ProjectionMode` + pure `project_spike_activity` / `spike_activity_features` (SNN→embedding for MoE; no learned W/b, no SAAQ) (#25).
- Pure MoE math in `routing`: synthetic gates, softmax, top-k, routing entropy; `SyntheticExpertRouter` under `backends` (#26).
- `ReverseHybridPath<R: ExpertRouter>` reverse-path orchestrator + integration tests
  (`tests/reverse_path.rs`); dual host alongside `HybridNetwork` (#23).
- `plan` module: typed `StageGraph` compiled into a validated
  `HybridExecutionPlan` — deterministic `StageId`s and topological order,
  explicit `Forward`/`Feedback` edges, `PortSpec`/`DimSpec` shape+dtype
  contracts with symbolic-dim unification, `ExecutionDomain` assignment hooks,
  structured `PlanError` variants (empty, duplicate names, unknown endpoints,
  disconnected, cyclic, unpermitted feedback, incompatible contract, invalid
  domain/contract, non-sequential stage IDs, invalid parameters),
  deterministic serde JSON (deserialization recompiles the full validation
  suite) and `describe()` inspection
  (Linear [RM-1804](https://linear.app/rpd-34/issue/RM-1804),
  [#40](https://github.com/rmems/hybrid-fusion/issues/40)).
- Compatibility paths: `HybridExecutionPlan::from_hybrid_config` /
  `from_reverse_path` describe the existing ANN→SNN and SNN→MoE flows as stage
  graphs; `HybridNetwork::execution_plan` / `ReverseHybridPath::execution_plan`
  expose them without changing numerical execution (RM-1804, #40).
- **BREAKING**: `HybridError` gains `ExecutionPlan(#[from] PlanError)`
  (exhaustive enum; downstream `match` arms must be updated) (RM-1804, #40).

### Changed

- **BREAKING (RM-1940):** `NeuromodSnn::new(inputs, outputs)` and
  `with_seed(inputs, outputs, seed)` require an explicit output population and
  return `Result`. The silent eight-neuron cap / `MAX_LIF_NEURONS` is removed.
  Zero dimensions and matrix payloads above `MAX_MATRIX_BYTES` (64 MiB) return
  `InvalidConfig` before allocation; `pre_inference_matrix_bytes` takes both
  dimensions. See [migration](docs/implementing-backends.md#rm-1940-migration).
- **BREAKING (RM-1940):** `HybridOutput` requires `num_neurons`, including in
  serialized records. `SpikingNetwork::num_neurons()` defaults to input width
  for existing one-output-per-input backends; unequal-width backends must
  override it. Hosts validate the output population/ID bounds and report it in
  outputs and execution plans. Reverse features use output population, never
  input width. Config-only plans now reject zero `snn_lif_neurons`.
- `NeuromodSnn::reset` also restores initial LIF thresholds (RM-1940).
  Upstream reset preserves reward-retuned thresholds; restoring only weights
  could change borderline fired IDs after reset/reseed versus a fresh run.
- Declared MSRV `rust-version = "1.98.1"` (required by neuromod 0.7). Cargo has
  no per-feature MSRV, so this applies to **all** builds, including the default
  dependency-light build, not only when the `neuromod` feature is enabled. The
  `neuromod` dependency and its transitive `rand` still stay out of the default
  build ([#44](https://github.com/rmems/hybrid-fusion/issues/44)).
- Demoted `SimpleSnn` (under the `backends` feature) to a reference/mock-only
  backend. Its behavior, code, and tests are unchanged; the documentation now
  states it is a deterministic mock for tests and examples, not a neuron-dynamics
  engine. Callers needing real dynamics should enable `neuromod` and use
  `NeuromodSnn` ([#44](https://github.com/rmems/hybrid-fusion/issues/44)).
- Docs and examples prefer `HybridNetwork::try_new`. `HybridNetwork::new`
  remains an unvalidated pre-1.0 compatibility wrapper and does not panic
  (Linear [RM-1357](https://linear.app/rpd-34/issue/RM-1357)).
- **`HybridNetwork::forward` preflights transformer hidden-state rank, sequence
  length, dimension, backing length, and finiteness**, and rejects a zero-channel
  SNN, before pooling or `snn.step`. Rank 2 `[seq, dim]` remains canonical; rank 1
  `[dim]` stays an explicit pre-pooled layout. Contract errors do not increment
  `global_step` and are not reported to Sentry ([#39](https://github.com/rmems/hybrid-fusion/pull/39)).
- **BREAKING**: `HybridError` gains `ConfigMismatch`, `SafetensorsParse`, and
  structured forward-contract variants. The enum is exhaustive (no
  `#[non_exhaustive]`); downstream `match` arms must be updated (RM-1357, #27,
  #39).
- **BREAKING**: `Dtype` now includes the full Safetensors header vocabulary (`U16`/`U32`/`U64`, `F4`, `F6_*`, `F8_*`, `C64`). Exhaustive matches must be updated (#27).
- **`HybridOutput` gains** optional `expert_weights`, `selected_experts`, `routing_entropy` (struct-literal / exhaustive matches must be updated). ANN→SNN `forward` sets them to `None` (#22).
- **`routing::softmax` accumulates its denominator in `f64`** (was `f32`). Returned
  weights now re-accumulate to `1.0` within one `f32` rounding at every expert
  count; previously they drifted by `O(n * 2^-24)` — up to `4.8e-2` at
  `MAX_REASONABLE_EXPERTS`. Weight *values* shift by at most one ULP (#23).
- **`ExpertRouteOutput` weight normalization is now enforced**: `ReverseHybridPath::forward_activity`
  rejects an `expert_weights` sum deviating from `1.0` by more than the new public
  `WEIGHT_SUM_TOLERANCE` (`8 * f32::EPSILON`) with `HybridError::InvalidConfig`;
  sums inside it are renormalized in `f64`. `ExpertRouter` implementors returning
  raw or partially normalized gate scores must normalize them, accumulating the
  denominator in `f64` (#23).
- Bump crate version `0.2.0` → `0.3.0` for the architecture-contracts milestone.
- Post-transfer hygiene: package `repository`, README CI badge, and docs now point at `rmems/hybrid-fusion` (#28).
- README sibling-crate links and ownership split clarified (pure MoE math in hybrid-fusion; tensor math in cortex-tensor; parse/mmap in engram-parser; dynamics in neuromod; runtime in brainstem-daemon).

### Fixed

- Large-width `NeuromodSnn` construction no longer implicitly allocates one
  fully connected LIF neuron per input channel (Linear
  [RM-1939](https://linear.app/rpd-34/issue/RM-1939)).
  [RM-1940](https://linear.app/rpd-34/issue/RM-1940) supersedes the interim
  eight-neuron cap with an explicit, configurable output population. Input
  width remains `num_channels()`; real fired IDs are in `0..num_neurons()`.
  Weights, eligibility traces, and replay weights use 16 × inputs × outputs
  bytes: linear in input width only for a fixed output population. At 16_384
  inputs and 8 outputs this is 2_097_152 bytes (2 MiB), not total process memory.
  Constructors reject matrix payloads above 64 MiB, including the former
  16_384 × 16_384 topology (4 GiB with 8-byte eligibility traces).
  `NeuromodSnn::pre_inference_matrix_bytes(inputs, outputs)` reports matrix
  payload only; neuron state, per-input state and allocator overhead are extra.

### Removed

- Qodana Cloud scan (`qodana-rust` + `QODANA_TOKEN`) after JetBrains membership expired (#35).

## [0.2.0] - 2026-07-10

First tagged release (`v0.2.0` → commit `796ca41`).

### Added

- `src/tensor.rs` — lightweight owned tensor type (data + shape).
- `src/traits.rs` — trait abstractions: `Transformer`, `SpikingNetwork`, `GgufLoader`, `NeuroModulators`.
- Optional `sentry` feature + reference `backends` feature and examples.
- Integration and property tests; Implementing a Backend guide; AGENTS.md; REVIEW.md.
- `LICENSE-MIT` and `LICENSE-APACHE` files.
- SPDX license headers on all `.rs` source files.
- GitHub Actions CI workflow (`.github/workflows/ci.yml`) with fmt, clippy, build, and test.
- This changelog.

### Changed

- **BREAKING**: Removed direct dependencies on `cortex-tensor`, `engram-parser`, and `neuromod`. The crate is now fully standalone and backend-agnostic.
- **BREAKING**: `HybridNetwork` is now generic over `Transformer` and `SpikingNetwork` traits.
- **BREAKING**: Replaced `cortex_tensor::Tensor` with a local `tensor::Tensor` type.
- **BREAKING**: `HybridConfig` now uses a local `TransformerConfig` instead of `cortex_tensor::transformer::TransformerConfig`.
- Migrated license from GPL-3.0-or-later to dual MIT/Apache-2.0.
- Updated README with scope/boundary documentation and dual-license badge.

### Removed

- Direct `cortex-tensor`, `engram-parser`, and `neuromod` crate dependencies.
- Old `rust.yml` CI workflow (replaced by `ci.yml`).
- GPL-3.0 `LICENSE` file (replaced by dual MIT/Apache-2.0).
