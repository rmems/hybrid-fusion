# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- `docs/extraction-map.md` — maps extractable LLM/SNN/MoE architecture from corinth-canal and grok-ozempic into hybrid-fusion traits vs sibling crates (#21).
- `src/tensor.rs` — lightweight owned tensor type (data + shape).
- `src/traits.rs` — trait abstractions: `Transformer`, `SpikingNetwork`, `GgufLoader`, `NeuroModulators`.
- `LICENSE-MIT` and `LICENSE-APACHE` files.
- SPDX license headers on all `.rs` source files.
- GitHub Actions CI workflow (`.github/workflows/ci.yml`) with fmt, clippy, build, and test.
- This changelog.

### Changed

- Post-transfer hygiene: package `repository`, README CI badge, and docs now point at `rmems/hybrid-fusion` (#28).
- README sibling-crate links updated for crates that live under `rmems` (neuromod remains under Limen-Neural until that return lands).
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
