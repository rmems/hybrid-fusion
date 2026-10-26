# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] — v0.3.0 track

Package version is **0.3.0** (start of the LLM/SNN architecture-contracts series).
Trait/API landings for epic [#20](https://github.com/rmems/hybrid-fusion/issues/20)
will accumulate here until a tagged `v0.3.0` GitHub release.

### Added

- **SNN step-contract sync with cortex-tensor** (Linear
  [RM-2360](https://linear.app/rpd-34/issue/RM-2360/snn-keep-cortex-tensors-snn-step-contract-in-sync-with-hybrid-fusions);
  design in [ADR 0002](docs/adr/0002-snn-step-contract-parity.md)):
  cortex-tensor keeps its own `src/snn` twin contract (decision RM-2206 —
  that crate is deliberately ANN+SNN capable; the removal plan
  [rmems/cortex-tensor#137](https://github.com/rmems/cortex-tensor/pull/137)
  / [#141](https://github.com/rmems/cortex-tensor/pull/141) was closed
  unmerged), so the two contracts are now held together explicitly instead
  of by prose.
  - New ADR 0002 documents the mapping between the trait sets (`SnnBackend`
    ↔ `SpikingNetwork`, `SnnEncoder`/`SnnDecoder` ↔
    `SpikeEncoder`/`SpikeDecoder`, `SpikingMoeRouter` ↔
    `SpikingExpertRouter`, `NeuromodNetwork` ↔ `NeuromodSnn`) and every
    intentional difference (width axes, decoder fallibility, reset
    semantics, output shape, capabilities vocabulary). It is mirrored in
    cortex-tensor as `docs/snn-contract-mapping.md`.
  - New parity tests
    `smoke/cortex-neuromod/tests/snn_contract_parity.rs` run the same seeded
    inputs through both spiking routers and both neuromod adapters in CI
    (validate + MSRV jobs): seeded identity-wired adapters must replay
    bit-for-bit, the boosted selection must agree, host validation must fail
    closed on out-of-range/duplicate fired IDs on both sides, and every
    intentional difference is asserted, not assumed.
  - `smoke/cortex-neuromod` pins cortex-tensor at
    `f591f2efd298e920250dd6cfdbf6056ea9368257` (past the
    [#91](https://github.com/rmems/cortex-tensor/issues/91)/
    [#92](https://github.com/rmems/cortex-tensor/issues/92)/
    [#93](https://github.com/rmems/cortex-tensor/issues/93) fix series) and
    enables its `neuromod` feature — the bump RM-2208 was waiting on; the
    smoke harness and the new parity tests pass against it.
  - Wording that claimed this crate is the *single* owner of the step
    contract, or that cortex-tensor "keeps no spiking types" (AGENTS.md,
    REVIEW.md, `docs/implementing-backends.md`, module docs, and the entry
    below) is corrected, and ADR 0001's single-owner decision is marked
    superseded by ADR 0002.
- **Single ANN↔SNN step contract** ([#61](https://github.com/rmems/hybrid-fusion/issues/61),
  Linear [RM-2205](https://linear.app/rpd-34/issue/RM-2205); design in ADR
  0001, [PR #64](https://github.com/rmems/hybrid-fusion/pull/64)): hybrid-fusion
  became an owner of the spiking step boundary. The cortex-tensor
  `src/snn` trait surface (`SnnBackend`, `SnnEncoder`, `SnnDecoder`,
  `SpikingMoeRouter`, `SnnCapabilities`, `SnnStepOutput`) was ported into this
  crate as its own twin contract; cortex-tensor keeps its `src/snn` copy
  (decision RM-2206/RM-2360 — the originally tracked removal was abandoned),
  and the cortex-tensor fixes for
  [#91](https://github.com/rmems/cortex-tensor/issues/91),
  [#92](https://github.com/rmems/cortex-tensor/issues/92), and
  [#93](https://github.com/rmems/cortex-tensor/issues/93) are ported with it,
  so both sides carry them and the parity tests hold them together.
  - New `encoding` module: `SpikeEncoder` / `SpikeDecoder` boundary traits with
    declared `SpikeEncoding` / `SpikeDecoding` metadata, plus `RateEncoder`
    and `SpikeCountDecoder` ported from cortex-tensor. `SpikeDecoder::decode`
    is fallible and sized by `num_neurons` (the output population), not the
    input channel count, and rejects out-of-range fired IDs instead of
    dropping them.
  - `SpikingNetwork::step_frozen` joins the step contract: a held-out tick
    that must not change plasticity-controlled state. The default **fails
    closed** (an unsupported-operation error, like the `reset` default), so a
    backend that does not override it can never silently run the learning path
    on a held-out call; overriders must advertise the `frozen_evaluation`
    feature. Ports cortex-tensor issue
    [#92](https://github.com/rmems/cortex-tensor/issues/92).
  - New `spiking_router` module: `SpikingExpertRouter<R, S, E, D>` — the
    spiking MoE routing host ported from cortex-tensor's `SpikingMoeRouter` —
    composing `ExpertRouter` gate logits, a `SpikeEncoder`, a
    `SpikingNetwork`, and a `SpikeDecoder` into gate → encode → step → decode
    → boosted top-k. `SpikingRouteOutput` carries the boosted route, the
    tick's fired IDs, the output population, and the bounded `hidden_gain`.
    `forward_frozen` checks the backend's `frozen_evaluation` capability
    before the encoder advances and never falls back to the learning path.
    Ports the cortex-tensor issue
    [#93](https://github.com/rmems/cortex-tensor/issues/93) fixes: additive
    spike boost (`gate + spike_boost × activity`, saturating at `f32::MAX`,
    `±Inf` gates keep their value; `with_spike_boost` validates finite ≥ 0),
    bounded hidden gain (selected softmax mass weighted by activity
    normalized to the tick peak; silent ticks keep the dense selected mass),
    and host-side validation of backend spikes and decoder readouts — with
    the ported tests.
  - `ExpertRouter::gate_logits(embedding) -> Result<Vec<f32>>`: pre-softmax
    logits, the scores `route` normalizes. The spike boost is defined on
    logits, not normalized weights, so `SpikingExpertRouter` requires it; the
    default fails closed and `StubExpertRouter` / `SyntheticExpertRouter`
    override it with their exact scores.
  - `NeuromodSnn::identity_wired(channels, seed)`: one LIF output neuron per
    input channel wired to its own channel at the configured maximum weight —
    the expert-*i*-at-neuron-*i* topology for `SpikingExpertRouter` — and
    drivable-wiring validation in `NeuromodSnn::from_network` (a LIF neuron
    with no strictly positive weight on an integrated channel, or a
    non-finite one, is rejected at construction instead of staying silent
    forever). Ports cortex-tensor issue
    [#91](https://github.com/rmems/cortex-tensor/issues/91).
  - `NeuromodSnn::step_frozen` routes through neuromod 0.7's
    `step_frozen_with_rng` (weights, eligibility, thresholds/decay, and stored
    modulators stay bit-for-bit unchanged; runtime state advances) and
    `capabilities()` advertises `frozen_evaluation`. Ports cortex-tensor
    issue [#92](https://github.com/rmems/cortex-tensor/issues/92).
  - `PortSpec` gains `encoding` / `decoding` (`Option`, `#[serde(default)]` so
    existing plans deserialize unchanged): the plan-side mirror of
    `SpikeEncoder::encoding` / `SpikeDecoder::decoding`. An edge into a
    consumer that declares an encoding/decoding must come from a producer
    declaring the same scheme and timestep/window count; a range mismatch is
    not an error (the projector already bounds stimuli).
- **BREAKING**: `HybridError` gains `SnnSpikeOutOfRange { neuron, num_neurons }`,
  `SnnDuplicateSpike { neuron }`, and `SnnInvalidReadout { index, value }`, and
  `ForwardValueStage` gains `EncoderStimulus` and `DecoderReadout` (both
  exhaustive enums; downstream `match` arms must be updated)
  ([#61](https://github.com/rmems/hybrid-fusion/issues/61)).
- **BREAKING**: `PortSpec` gains `encoding` / `decoding` fields (struct
  literals must be updated; the `any` / `f32` / `f32_exact` constructors and
  deserialization keep the old behavior)
  ([#61](https://github.com/rmems/hybrid-fusion/issues/61)).
- **BREAKING**: `HybridNetwork::forward` reports a fired ID outside the output
  population as `HybridError::SnnSpikeOutOfRange` (was free-text
  `HybridError::SnnStep`), and additionally rejects a backend reporting the
  same neuron twice in one tick with `HybridError::SnnDuplicateSpike`. Spike
  validation is host-side, so every `SpikingNetwork` is checked, not only the
  spiking-router path ([#61](https://github.com/rmems/hybrid-fusion/issues/61),
  cortex-tensor [#93](https://github.com/rmems/cortex-tensor/issues/93)).
- Format-neutral planning topology: `AnnTopology`, `SnnTopology`,
  `HybridTopology`, and `TopologySource`, plus
  `HybridExecutionPlan::from_topology` and `from_reverse_topology`. Planning
  reads hidden width, maximum sequence length, SNN input channels, and SNN
  output population. `from_reverse_topology` takes the router embedding width
  separately from ANN hidden width. It has no checkpoint-format dependency. Programmatic and
  mock topology need no checkpoint file and no engram-parser crate.
  `TopologySource` is the integration point for a future engram-parser
  topology adapter. `AnnTopology.architecture` is descriptive and does not
  change the compiled plan
  ([#42](https://github.com/rmems/hybrid-fusion/issues/42)).
- **BREAKING**: `PlanError` gains `InvalidTopology { field, reason }`
  (exhaustive enum; downstream `match` arms must be updated). `field` is a
  stable discriminator (`hidden_dim`, `max_seq_len`, `input_channels`,
  `output_population`), matched the same way as `IncompatibleCapability.field`.
  `HybridError::ExecutionPlan` carries the variant
  ([#42](https://github.com/rmems/hybrid-fusion/issues/42)).
- `BackendCapabilities` plus `CapabilityNegotiation`: ANN and SNN traits expose
  a side-effect-free capability report, and a compiled `HybridExecutionPlan`
  can be checked against offered backends before execution. Unsupported
  domain, dtype, width, output population, sequence, batch, streaming,
  stateful, reset, and feature pairings return structured `PlanError` variants.
  A requirement must match the stage's resolved domain and every dtype or fixed
  dimension derivable from its port contract or canonical dimension metadata.
  Fallback runs only under
  `FallbackPolicy::AllowNamed` and only for a backend the caller names,
  including when the preferred backend was not offered; `Forbid` never
  substitutes. Empty backend identities are rejected. The default SNN report
  advertises `step` and a one-step temporal window without claiming cross-call
  state. Reset is one typed capability (`requires_reset` / `reset`); the named
  `"reset"` feature is normalized to it.
  `NeuromodSnn::capabilities` reports `backend_name: "neuromod::SpikingNetwork"`
  with the same flags cortex-tensor's adapter publishes (caller RNG, plasticity,
  neuromodulation; frozen evaluation absent). Linear RM-1421 produced no audit
  findings — its only session died on a GitHub rate limit — so there was no
  finding list to port ([#41](https://github.com/rmems/hybrid-fusion/issues/41)).
- **BREAKING**: `PlanError` gains `UnsupportedBackend`, `IncompatibleCapability`,
  `UnsupportedFeature`, and `SemanticMismatch` (exhaustive enum; downstream
  `match` arms must be updated) ([#41](https://github.com/rmems/hybrid-fusion/issues/41)).
- `SpikingNetwork::reset` provides the polymorphic operation promised by the
  reset capability. Its default returns an unsupported-operation error;
  `SimpleSnn` and `NeuromodSnn` override it and advertise reset support.
- `smoke/cortex-neuromod`: a separately resolved harness (not a workspace
  member, `publish = false`) that runs one real cortex-tensor
  `ReferenceExecutor` embedding stage, projected with
  `embed_to_stimuli_with_width`, into `NeuromodSnn::with_seed`. Fixed seed
  `0x45_5EE5`, 4 channels, 4 steps. Asserts finite hidden state and stimuli,
  in-range fired indices, a spike total in `1..=16`, and identical fired
  indices across two same-seed runs and after `reset()` + `reseed`. A different
  seed may differ. cortex-tensor is not on crates.io; the harness pins git rev
  `5cbee81473ed69bd8e74d53e210ef64c02d232da` in its own `Cargo.lock`. It is not
  a library feature, so default and `neuromod`-only resolution do not fetch it
  ([#45](https://github.com/rmems/hybrid-fusion/issues/45)).
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

### Deprecated

- `GgufLoader`, `GgufLayout`, `SafetensorsLoader`, and `SafetensorsLayout` are
  deprecated since 0.4.0. Signatures, public fields, `SafetensorsLayout::new`,
  ordered `entries_with_role`, deserialization count normalization, and serde
  wire behavior are unchanged. Parse checkpoints with engram-parser. Pass
  `AnnTopology` or `HybridTopology` into planning. SNN topology stays
  caller-supplied. Removal is a later minor release, no earlier than the
  release after engram-parser publishes its topology API. See
  [Migrating off format-specific loaders](docs/implementing-backends.md#migrating-off-format-specific-loaders).
  `TensorRole`, `TensorManifestEntry`, and `Dtype` stay as legacy inventory
  helpers and are not deprecated
  ([#42](https://github.com/rmems/hybrid-fusion/issues/42)).

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
- Declared MSRV `rust-version = "1.99.0"`, matching `rust-toolchain.toml`,
  the smoke harness, and the CI MSRV job (Linear RM-2050). neuromod 0.7
  originally required 1.98.1; Cargo has no per-feature MSRV, so the floor
  applies to **all** builds, including the default dependency-light build.
  The `neuromod` dependency and its transitive `rand` still stay out of the
  default build ([#44](https://github.com/rmems/hybrid-fusion/issues/44)).
- Package `exclude` now drops agent, CI, and review-only paths
  (`.agents/`, `.github/`, `.coderabbit.yaml`, `AGENTS.md`, `REVIEW.md`,
  `rust-toolchain.toml`, `docs/superpowers/`, `smoke/`) and keeps
  `docs/implementing-backends.md` and `docs/extraction-map.md` in the
  published crate.
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

- **BREAKING (reverse path / RM-1445):** a `SpikeActivity` timestep is
  defined as a **set**, matching the forward host's at-most-once fired-ID
  contract from [#61](https://github.com/rmems/hybrid-fusion/issues/61)
  (`HybridError::SnnDuplicateSpike`). `SpikeActivity::from_fired` and
  `spike_activity_features` (so also `project_spike_activity` and
  `ReverseHybridPath::forward_activity`) now **reject** an index that appears
  more than once in one step with `HybridError::InvalidConfig` naming the
  neuron and step, instead of counting it into the rate. Previously three
  copies of neuron `0` in a single step produced a `RateSum` rate of `3.0`,
  which then silently saturated through the projector's `tanh`. Multisets are
  not a supported input; callers holding one must deduplicate first. With
  uniqueness enforced, `RateSum` / `SpikingTernary` features are per-neuron
  means over steps and are guaranteed to stay in `[0, 1]` by construction
  (documented on `spike_activity_features` and `firing_rates`; the `tanh` is
  a squash of an already-bounded vector, not a substitute for the contract).
- **BREAKING (reverse path / RM-1445):** `SpikeActivity::iz_potentials` now
  has a **defined contract** instead of being silently ignored at any length.
  No `ProjectionMode` reads it (the adaptive/Izhikevich-GIF bank lives in
  `neuromod`; ternary/GIF events are a non-extract per
  `docs/extraction-map.md`), so the field is kept for trace/serde fidelity
  only: **empty** is the unused spelling, **`len == n_neurons`** with finite
  values is accepted and documented as ignored by every pure mode, and any
  other length is rejected with `HybridError::InputLengthMismatch` while a
  non-finite value is rejected with `InvalidConfig` — the 99-element-vector
  case now fails instead of being dropped. Enforced in
  `spike_activity_features`, hence through `project_spike_activity`,
  `ReverseHybridPath::forward_activity`, and `SpikingTernary`/`RateSum`/
  `TemporalHistogram`/`MembraneSnapshot` alike.
- `ProjectionMode::SpikingTernary` is doc-commented as an **alias** of
  `RateSum` (issue #25) with `#[doc(alias = "RateSum")]`: it emits identical
  firing-rate features, encodes no ternary (spike/burst/silent) events, and
  reads no `iz_potentials` until a neuromod GIF mode lands
  ([#25](https://github.com/rmems/hybrid-fusion/issues/25),
  Linear [RM-1445](https://linear.app/rpd-34/issue/RM-1445)).
- **BREAKING (serde / RM-1465):** JSON serialization of the `f32`-bearing record
  types now **fails closed on non-finite values** instead of writing `null`
  (Linear [RM-1465](https://linear.app/rpd-34/issue/RM-1465)). JSON is not a
  lossless `f32` container: `serde_json` has no spelling for `NaN` or ±`Inf`
  and silently emits `null`, so `HybridOutput { embedding: vec![NaN], … }`
  serialized to `"embedding":[null]` and `routing_entropy: Some(NaN)` to
  `"routing_entropy":null`, which reads back as `None` — silently wrong logs,
  Sentry breadcrumbs, and checkpointed fusion traces. `HybridOutput`
  (`embedding`, `stimuli`, `expert_weights`, `routing_entropy`),
  `ExpertRouteOutput`, `SpikingRouteOutput` (`hidden_gain`, plus the nested
  route's own guard), `SpikeActivity` (`potentials`, `iz_potentials`),
  `NeuroModulators`, and `Tensor` now return a serde error naming the offending
  field, index, and value (e.g. `HybridOutput.embedding[1] is NaN: …`); via
  `serde_json` that is a `serde_json::Error`, which `HybridError::Json` wraps
  with `?`. Deserialization is unchanged: strict JSON cannot carry a
  non-finite number in the first place, `null` stays the wire spelling of
  `None` for the optional fields, and legacy records containing `null` inside
  a float array are already rejected by `serde_json` (`invalid type: null,
  expected f32`) — a legacy `"routing_entropy": null` must be re-derived, not
  trusted. Finite records round-trip losslessly; new tests cover the tiny
  forward-path and reverse-path outputs plus every guarded field. Complements
  the runtime NaN rejection in `HybridNetwork::forward` (RM-1356), which cannot
  see a router that reports non-finite `routing_entropy` or a hand-built
  `HybridOutput`.
- **BREAKING**: `routing::softmax` now returns `Result<Vec<f32>>` and has a
  documented non-finite policy. `NaN` and `+Inf` scores no longer produce a
  uniform distribution. A single `+Inf` score gives a one-hot result, several `+Inf`
  scores split the mass equally among only those indices, and any `NaN` returns
  `HybridError::InvalidConfig`. All `-Inf` scores still give a uniform result,
  which is the limit for equal scores. Previously `softmax([+Inf, 0, 0])` and
  `softmax([NaN, 1])` both returned a uniform distribution.
  `route_synthetic` and `SyntheticExpertRouter::route` reject `NaN` gate scores
  (from a `NaN` embedding element, or `+Inf` and `-Inf` in one chunk) instead of
  routing uniformly (Linear [RM-1442](https://linear.app/rpd-34/issue/RM-1442)).
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
