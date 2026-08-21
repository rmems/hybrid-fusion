# AGENTS.md — hybrid-fusion

**Version:** 0.3.0 (last updated 2026-08-11)

Context file for AI coding agents working in this repository.

## Agent identity

You are an AI coding agent working on **hybrid-fusion** under the **rmems**
GitHub account (returned from Limen-Neural). Prefer small, test-backed changes
that respect crate boundaries. Do not invent concrete backend dependencies in
this crate.

## Tools available

Typical local tools for this repo:

| Tool | Purpose |
|------|---------|
| `cargo build` / `cargo test --all-features` | Compile and run unit tests |
| `cargo clippy -- -D warnings` | Lint at CI severity |
| `cargo fmt` | Format Rust sources |
| `gh` | GitHub PR / CI inspection (optional) |
| Editor / language server | Navigate `src/traits.rs` and sibling crates |

Run `cargo test --all-features` before pushing CI-sensitive changes.

## Crate purpose

`hybrid-fusion` is a **backend-agnostic pure-Rust orchestrator** for hybrid
Transformer ↔ Spiking Neural Network (SNN) forward-pass pipelines. It depends
only on trait abstractions — no concrete math, neuron dynamics, or model-loading
implementations live here.

## Crate boundaries

### This crate owns

- Orchestration of hybrid artificial neural network (ANN) → SNN forward-pass paths
- Transformer hidden-state pooling / resizing into bounded SNN stimuli
- The public `HybridNetwork<T, S>` API and its error boundaries
- Projector logic (dimensionality reduction from transformer hidden space to SNN input channels)
- Configuration and output types (`HybridConfig`, `TransformerConfig`, `HybridOutput`)
- Reverse-path host `ReverseHybridPath` (activity → embedding → ExpertRouter)

### This crate does not own

| Concern | Owned by |
|---|---|
| Tensor / transformer math | `cortex-tensor` |
| GGUF (GPT-Generated Unified Format) model parsing | `engram-parser` |
| SNN neuron dynamics (leaky integrate-and-fire (LIF), Izhikevich) | `neuromod` |
| SNN runtime / scheduling | `brainstem-daemon` |
| Neuromodulator mapping / critic signals | `limbic-critic` |
| Sensory encoding | `axon-encoder` |

**Default rule (with escape hatch):** Prefer trait-only interaction with backends.
Do not add concrete backend implementations to this crate's `Cargo.toml` unless a
tracked issue explicitly authorizes a temporary exception and documents removal.
All normal backend interaction flows through the trait contracts in `src/traits.rs`.

## Trait contracts

Core traits / types on the pluggable surface:

- **`Transformer`** — produces hidden-state tensors from token IDs. Implementations live in downstream crates (e.g. `cortex-tensor`).
- **`SpikingNetwork`** — steps the SNN forward given stimuli + neuromodulator state, returns fired neuron indices. Implementations live in `neuromod` / `brainstem-daemon`.
- **`GgufLoader`** — loads GGUF model layouts from disk. Implementations live in `engram-parser`.
- **`ExpertRouter`** + **`SpikeActivity`** — reverse-path MoE routing contract (embedding → expert weights / selection). Pure math lives in `src/routing.rs`; checkpoint backends stay outside this crate.

`HybridNetwork<T: Transformer, S: SpikingNetwork>` is generic over the `Transformer` and `SpikingNetwork` traits; `GgufLoader` / `ExpertRouter` are consumed separately. Adding a concrete backend dependency to `Cargo.toml` is a boundary violation unless covered by the escape hatch above.

`ReverseHybridPath<R: ExpertRouter>` is the dual reverse-path host (SNN activity →
project → MoE); do not fold it into `HybridNetwork` generics.

## Sibling crates

| Crate | Responsibility |
|---|---|
| `neuromod` | SNN neuron dynamics (LIF, Izhikevich models) |
| `cortex-tensor` | Tensor math and transformer inference |
| `engram-parser` | GGUF file parsing and model layout extraction |
| `brainstem-daemon` | SNN runtime scheduling and execution |
| `limbic-critic` | Neuromodulator-to-critic signal mapping |
| `axon-encoder` | Sensory input encoding |

## Build & test

```bash
cargo build
cargo test --all-features
cargo clippy -- -D warnings
```

## Extraction map (v0.3 planning)

Before porting new LLM/SNN/MoE traits, read
[docs/extraction-map.md](docs/extraction-map.md). It maps research modules from
corinth-canal and grok-ozempic to hybrid-fusion contracts vs sibling crates, and
lists non-extract items (SAAQ, mmap parsers, CUDA, GOZ1).

## Org boundary matrix

See [hybrid-fusion issue #5](https://github.com/rmems/hybrid-fusion/issues/5)
for cross-crate ownership and dependency rules.
