# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] — v0.3.0 track

Package version is **0.3.0** (start of the LLM/SNN architecture-contracts series).
Trait/API landings for epic [#20](https://github.com/rmems/hybrid-fusion/issues/20)
will accumulate here until a tagged `v0.3.0` GitHub release.

### Added

- `docs/extraction-map.md` — maps extractable LLM/SNN/MoE architecture from corinth-canal and grok-ozempic into hybrid-fusion traits vs sibling crates (#21).
- Reverse-path contracts: `SpikeActivity`, `ExpertRouter`, `ExpertRouteOutput`; MoE fields on `HybridOutput`; `StubExpertRouter` under `backends` (#22).

### Changed

- Bump crate version `0.2.0` → `0.3.0` for the architecture-contracts milestone.
- Post-transfer hygiene: package `repository`, README CI badge, and docs now point at `rmems/hybrid-fusion` (#28).
- README sibling-crate links and ownership split clarified (pure MoE math in hybrid-fusion; tensor math in cortex-tensor; parse/mmap in engram-parser; dynamics in neuromod; runtime in brainstem-daemon).

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
