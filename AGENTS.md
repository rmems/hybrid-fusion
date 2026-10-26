# AGENTS.md — hybrid-fusion

**Version:** 0.3.0 (last updated 2026-10-26)

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

- **hybrid-fusion's ANN↔SNN step contract** (issue #61 / RM-2205): the
  `SpikingNetwork` step traits plus the `SpikeEncoder` / `SpikeDecoder`
  boundary adapters in `src/encoding.rs`. cortex-tensor keeps its own twin
  `src/snn` contract (`SnnBackend` / `SnnEncoder` / `SnnDecoder`,
  `SpikingMoeRouter`, `NeuromodNetwork`) because that crate is deliberately
  ANN+SNN capable (decision RM-2206 / RM-2360; the removal plan
  rmems/cortex-tensor#137 / #141 was abandoned unmerged). The two trait sets
  are held in sync by the mapping in
  [docs/adr/0002-snn-step-contract-parity.md](docs/adr/0002-snn-step-contract-parity.md)
  and by the cross-crate parity tests in
  `smoke/cortex-neuromod/tests/snn_contract_parity.rs`. Never claim
  cortex-tensor is ANN-only or that this contract is the stack's single one.
- Spiking MoE routing host `SpikingExpertRouter` (gate logits → encode → step
  → decode → boosted top-k) in `src/spiking_router.rs`
- Orchestration of hybrid artificial neural network (ANN) → SNN forward-pass paths
- Transformer hidden-state pooling / resizing into bounded SNN stimuli
- The public `HybridNetwork<T, S>` API and its error boundaries
- Projector logic (dimensionality reduction from transformer hidden space to SNN input channels)
- Configuration and output types (`HybridConfig`, `TransformerConfig`, `HybridOutput`)
- Reverse-path host `ReverseHybridPath` (activity → embedding → ExpertRouter)
- Neutral planning topology (`AnnTopology`, `SnnTopology`, `HybridTopology`) and execution plans compiled from it

### This crate does not own

| Concern | Owned by |
|---|---|
| Tensor / transformer math | `cortex-tensor` |
| GGUF / Safetensors parsing, mmap, layout discovery, tensor roles, and topology discovery | `engram-parser` (sole checkpoint-format owner) |
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
- **`SpikingNetwork`** — hybrid-fusion's ANN↔SNN step contract (issue #61; cortex-tensor keeps the twin `SnnBackend` — mapping and parity gate in [ADR 0002](docs/adr/0002-snn-step-contract-parity.md)): `step` (learning tick) and `step_frozen` (held-out tick; default fails closed) advance the SNN given stimuli + neuromodulator state and return fired neuron indices in `0..num_neurons()`. Implementations live in `neuromod` (via the `NeuromodSnn` adapter) / `brainstem-daemon`. Hosts validate fired-ID bounds and at-most-once per tick.
- **`SpikeEncoder` / `SpikeDecoder`** — ANN↔SNN boundary adapters (`src/encoding.rs`, ported from cortex-tensor): ANN signal → stimuli and fired IDs → readouts, with declared `SpikeEncoding` / `SpikeDecoding` metadata mirrored onto `PortSpec`.
- **`SpikingExpertRouter`** — hybrid spiking MoE routing host (`src/spiking_router.rs`, ported from cortex-tensor's `SpikingMoeRouter` with the #93 fixes): gate logits → encode → step → decode → additive spike boost → top-k. A host like `ReverseHybridPath`, not a `HybridNetwork` generic.
- **`ExpertRouter`** + **`SpikeActivity`** — reverse-path MoE routing contract (embedding → expert weights / selection). `gate_logits` exposes the pre-softmax scores the spike boost needs (default fails closed). Pure math lives in `src/routing.rs`; checkpoint backends stay outside this crate.
- **`AnnTopology`** / **`SnnTopology`** / **`HybridTopology`** / **`TopologySource`** — format-neutral planning input. ANN dimensions may come from a `TopologySource`. SNN dimensions are caller-supplied. This crate does not open checkpoint files.
- **`GgufLoader`** / **`SafetensorsLoader`** — deprecated compatibility shims. Checkpoint parsing, mmap, layout discovery, tensor roles, and topology discovery belong to `engram-parser`. Do not add a new loader abstraction here. Migration: [docs/implementing-backends.md](docs/implementing-backends.md#migrating-off-format-specific-loaders).

`HybridNetwork<T: Transformer, S: SpikingNetwork>` is generic over the `Transformer` and `SpikingNetwork` traits. Planning consumes `HybridTopology` and does not require a GGUF or Safetensors loader. `ExpertRouter` is consumed separately. Adding a concrete backend dependency to `Cargo.toml` is a boundary violation unless covered by the escape hatch above.

`ReverseHybridPath<R: ExpertRouter>` is the dual reverse-path host (SNN activity →
project → MoE); do not fold it into `HybridNetwork` generics.

## Sibling crates

| Crate | Responsibility |
|---|---|
| `neuromod` | SNN neuron dynamics (LIF, Izhikevich models) |
| `cortex-tensor` | Tensor math and transformer inference |
| `engram-parser` | Sole checkpoint-format owner: GGUF and Safetensors parsing, mmap, layout discovery, tensor roles, and topology discovery |
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
