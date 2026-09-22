# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] — v0.3.0 track

Package version is **0.3.0** (start of the LLM/SNN architecture-contracts series).
Trait/API landings for epic [#20](https://github.com/rmems/hybrid-fusion/issues/20)
will accumulate here until a tagged `v0.3.0` GitHub release.

### Added

- Structured forward-path contract errors: `HiddenStateRank`, `HiddenStateSeqLen`,
  `HiddenStateDim`, `HiddenStateDataLen`, `NonFinite { stage, index }` (with
  `ForwardValueStage`), and `ZeroSnnChannels` (RM-1356).
- `docs/extraction-map.md` — maps extractable LLM/SNN/MoE architecture from corinth-canal and grok-ozempic into hybrid-fusion traits vs sibling crates (#21).
- Dual checkpoint contract: `SafetensorsLoader` / `SafetensorsLayout` / `TensorManifestEntry`, plus `Dtype` (full Safetensors header vocabulary), `TensorRole` (name-heuristic classifier), and `HybridError::SafetensorsParse` ([#27](https://github.com/rmems/hybrid-fusion/issues/27)).
- Reverse-path contracts: `SpikeActivity`, `ExpertRouter`, `ExpertRouteOutput`; MoE fields on `HybridOutput`; `StubExpertRouter` under `backends` (#22).
- `ProjectionMode` + pure `project_spike_activity` / `spike_activity_features` (SNN→embedding for MoE; no learned W/b, no SAAQ) (#25).
- Pure MoE math in `routing`: synthetic gates, softmax, top-k, routing entropy; `SyntheticExpertRouter` under `backends` (#26).
- `ReverseHybridPath<R: ExpertRouter>` reverse-path orchestrator + integration tests
  (`tests/reverse_path.rs`); dual host alongside `HybridNetwork` (#23).

### Changed

- **`HybridNetwork::forward` preflights transformer hidden-state rank, sequence
  length, dimension, backing length, and finiteness**, and rejects a zero-channel
  SNN, before pooling or `snn.step`. Rank 2 `[seq, dim]` remains canonical; rank 1
  `[dim]` stays an explicit pre-pooled layout. Contract errors do not increment
  `global_step` and are not reported to Sentry (RM-1356).
- **BREAKING**: `HybridError` gains `SafetensorsParse`. The enum is exhaustive (no `#[non_exhaustive]`); downstream `match` arms must be updated (#27).
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
