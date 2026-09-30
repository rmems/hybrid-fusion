# hybrid-fusion

[![CI](https://github.com/rmems/hybrid-fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/rmems/hybrid-fusion/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

**Pure-Rust master orchestrator for a hybrid transformer <-> spiking neural
network stack.** Zero Candle, zero CUDA, zero Julia.

`hybrid-fusion` defines the orchestration contract for piping transformer
hidden states into spiking neural network dynamics. It is intentionally
**backend-agnostic**: transformer and SNN implementations are injected via
traits (`Transformer`, `SpikingNetwork`), so downstream consumers choose
their own math engines.

## Architecture

```text
token_ids: &[u32]
     |
     v  Transformer::hidden_states
Tensor   [seq_len, dim]
     |
     v  projector::embed_to_stimuli_with_width   (pool -> resize -> tanh)
stimuli: Vec<f32>       in [-1, 1], length == snn.num_channels()
     |
     v  SpikingNetwork::step(&stimuli, &modulators)
fired_neurons: Vec<usize>
```

The `tanh` squash is applied **after** pooling and resizing so the values
fed into the SNN are always bounded.

## Scope / Boundaries

This crate **owns**:

- Orchestration of hybrid ANN -> SNN forward-pass paths.
- Transformer hidden-state pooling and resizing into bounded SNN stimuli.
- The public `HybridNetwork<T, S>` API and error boundaries.
- Pure MoE routing math (top-k, normalize, synthetic gate scores) in `routing`
  ([#26](https://github.com/rmems/hybrid-fusion/issues/26)); not tensor matmul
  against checkpoint weights.
- Reverse-path orchestration (`ReverseHybridPath`): SNN activity → embedding → MoE route
  ([#23](https://github.com/rmems/hybrid-fusion/issues/23)).

This crate **does not own**:

- Tensor / transformer math -> [`cortex-tensor`](https://github.com/rmems/cortex-tensor)
  (including real-weight gate matmul once backends land).
- GGUF / Safetensors parse, mmap, and layout discovery ->
  [`engram-parser`](https://github.com/rmems/engram-parser).
- Neuron dynamics and SNN integration internals ->
  [`neuromod`](https://github.com/Limen-Neural/neuromod)
  (still under Limen-Neural until that crate returns to `rmems`).
- SNN runtime / scheduling -> `brainstem-daemon`.

See [issue #5](https://github.com/rmems/hybrid-fusion/issues/5) for the full
boundary matrix, and [docs/extraction-map.md](docs/extraction-map.md) for the
v0.3 ownership plan.

## Quick start

Prefer [`HybridNetwork::try_new`](https://docs.rs/hybrid-fusion/latest/hybrid_fusion/struct.HybridNetwork.html#method.try_new)
so transformer dim, max sequence length, and SNN channel count are checked
against the backends before the first forward call.
[`HybridNetwork::new`](https://docs.rs/hybrid-fusion/latest/hybrid_fusion/struct.HybridNetwork.html#method.new)
remains as an unvalidated pre-1.0 compatibility constructor.

```rust
use hybrid_fusion::{
    HybridConfig, HybridNetwork, NeuroModulators, Result, SpikingNetwork, Tensor, Transformer,
};

struct TinyTransformer {
    dim: usize,
    max_seq_len: usize,
}

impl Transformer for TinyTransformer {
    fn hidden_states(&self, token_ids: &[u32]) -> Tensor {
        let seq = token_ids.len();
        Tensor::from_vec(vec![0.1; seq * self.dim], &[seq, self.dim])
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn param_count(&self) -> usize {
        0
    }
}

struct TinySnn {
    channels: usize,
}

impl SpikingNetwork for TinySnn {
    fn step(
        &mut self,
        _stimuli: &[f32],
        _modulators: &NeuroModulators,
    ) -> Result<Vec<usize>> {
        Ok(Vec::new())
    }
    fn num_channels(&self) -> usize {
        self.channels
    }
}

let config = HybridConfig::tiny();
let transformer = TinyTransformer {
    dim: config.transformer.dim,
    max_seq_len: config.transformer.max_seq_len,
};
let snn = TinySnn {
    channels: config.snn_input_channels,
};
let mut net = HybridNetwork::try_new(transformer, snn, config).unwrap();
let out = net.forward(&[1u32, 2, 3, 4], None).unwrap();
assert_eq!(out.embedding.len(), 128);
```

## Public surface

| Item | Purpose |
|------|---------|
| `HybridNetwork<T, S>` | Generic orchestrator over any `Transformer` + `SpikingNetwork`. Prefer `try_new` for construction. |
| `ReverseHybridPath<R>` | Reverse-path host: activity → project → `ExpertRouter` → MoE fields. |
| `Transformer` trait | Backend-agnostic transformer interface. |
| `SpikingNetwork` trait | Backend-agnostic SNN interface. |
| `ExpertRouter` trait | MoE routing: embedding → expert weights / selection. |
| `SpikeActivity` | Pure spike/membrane bag for reverse-path projection. |
| `ProjectionMode` + `project_spike_activity` | SNN activity → dense embedding for MoE (pure modes). |
| `routing` helpers | Always-on pure MoE math (gates, softmax, top-k, entropy). |
| `GgufLoader` / `SafetensorsLoader` | Dual checkpoint **layout** contracts (parse/mmap → `engram-parser`). |
| `SyntheticExpertRouter` / `StubExpertRouter` | File-free `ExpertRouter` impls (**requires `backends` feature**). |
| `NeuromodSnn` | Real SNN backend adapter over `neuromod` 0.7 (LIF/Izhikevich) implementing `SpikingNetwork` (**requires `neuromod` feature**). `SimpleSnn` under `backends` is a reference/mock only. |
| `NeuroModulators` | Neuromodulator struct passed to SNN steps. |
| `HybridConfig` / `TransformerConfig` | Predefined configs (`tiny`, `olmo_1b`). |
| `projector::embed_to_stimuli_with_width` | Pool -> resize -> tanh adapter. |
| `Tensor` | Lightweight owned tensor (data + shape). |
| `HybridError` / `ForwardValueStage` | Orchestration errors, including hidden-state contract diagnostics. |

## Guides

- **[Implementing a Backend](docs/implementing-backends.md)** — trait contracts, data flow, tensor shape conventions, and a minimal working example for `Transformer` + `SpikingNetwork`.
- **[Extraction map](docs/extraction-map.md)** — what is extractable from corinth-canal / grok-ozempic into hybrid-fusion vs sibling crates (MoE, dual GGUF+Safetensors, non-extract list).

## Spiking backend (optional `neuromod` feature)

`hybrid-fusion` stays backend-agnostic, but it ships one production-oriented
`SpikingNetwork` implementation behind the optional `neuromod` feature:
`NeuromodSnn`, an adapter over [`neuromod`](https://crates.io/crates/neuromod)
0.7's LIF/Izhikevich spiking engine. `SimpleSnn` (under the `backends` feature)
is a deterministic reference/mock only, kept for tests and examples; enable
`neuromod` and use `NeuromodSnn` when you need real neuron dynamics.

```sh
cargo build --features neuromod
```

The crate's MSRV is `rust-version = 1.98.1` for **all** builds, default and
feature-enabled alike. Cargo applies `rust-version` package-wide and has no
per-feature MSRV, so the bump (required by neuromod 0.7) applies even to the
default, dependency-light build. The `neuromod` dependency (and its transitive
`rand`) stays out of the default build; only the toolchain floor is shared.

### Neuromodulator mapping

`neuromod`'s modulator vocabulary (`{ dopamine, serotonin, acetylcholine,
norepinephrine }`, all defaulting to `0.0`) differs from hybrid-fusion's.
`NeuromodSnn` converts explicitly, setting every neuromod field:

| hybrid-fusion field | neuromod field   | note |
|---------------------|------------------|------|
| `dopamine`          | `dopamine`       | `(dopamine + aux_dopamine).clamp(0.0, 1.0)` |
| `aux_dopamine`      | `dopamine`       | folded into `dopamine` (secondary reward channel) |
| `cortisol`          | `norepinephrine` | direct; norepinephrine models stress/arousal |
| `acetylcholine`     | `acetylcholine`  | direct |
| `tempo`             | *(none)*         | intentionally not mapped (step/timebase policy; neuromod's step has no timebase input) |
| *(none)*            | `serotonin`      | left at `0.0`; no hybrid-fusion source |

Note that hybrid-fusion defaults `dopamine` and `acetylcholine` to `0.5` and
`tempo` to `1.0`, while all neuromod modulators default to `0.0`.

Stepping is deterministic by construction: build with
`NeuromodSnn::with_seed(num_channels, seed)`, and the trait `step` routes
through an owned seeded generator (backed by neuromod's `step_with_rng`), so the
same seed and stimulus sequence reproduce the same fired-index sequence.
`reset()` resets neuron dynamics without reseeding; `reseed(seed)` restarts the
random stream.

### Public vocabulary decision

hybrid-fusion keeps its public `NeuroModulators { dopamine, cortisol,
acetylcholine, tempo, aux_dopamine }` unchanged, and the adapter performs the
explicit conversion above. This keeps churn low, preserves backward
compatibility for existing callers, and keeps neuromod types out of the public
contract: no neuromod type appears in any public signature or re-export.

## Error monitoring (optional)

`hybrid-fusion` ships an optional
[Sentry](https://docs.sentry.io/platforms/rust/) integration. When the
`sentry` feature is enabled:

1. The `sentry` crate is **re-exported** as `hybrid_fusion::sentry` so
   applications share the same crate version and global hub as the library.
2. `HybridNetwork::forward` captures **backend/runtime** failures (e.g. SNN
   step errors) via `telemetry::capture_error`. Caller validation errors
   (`InputLengthMismatch`, `ConfigMismatch`, other config mismatches) are
   returned without capture so routine bad requests do not flood Sentry quota.
3. Panic capture is enabled through the underlying Sentry client features.
   Apps can also call `hybrid_fusion::telemetry::capture_error` for their
   own error paths.

**Note:** enabling `sentry` pulls a sizable transitive dependency tree
(reqwest/hyper/tokio/ring). Prefer it in service binaries, not lean library
builds.

### Enabling

```sh
cargo build --features sentry
# or, with the reference backends:
cargo build --features "sentry,backends"
```

### Configuration

Set the `SENTRY_DSN` environment variable to your Sentry project DSN.
**Do not commit DSNs** — pass them at runtime or via a secrets manager.

```sh
export SENTRY_DSN=https://examplePublicKey@o0.ingest.sentry.io/0
```

Sentry's Rust SDK reads `SENTRY_DSN` automatically; an empty/missing DSN
disables transport (safe no-op for local dev).

### Initialisation pattern

Always initialise through the **re-export** so events raised inside
`hybrid-fusion` land on the same hub:

```rust
// Requires: hybrid-fusion = { version = "0.3", features = ["sentry"] }
let _guard = hybrid_fusion::sentry::init((
    // Empty string / missing env → client is disabled (no network).
    std::env::var("SENTRY_DSN").unwrap_or_default(),
    hybrid_fusion::sentry::ClientOptions {
        release: hybrid_fusion::sentry::release_name!(),
        ..Default::default()
    },
));
```

The `_guard` must be held for the lifetime of the application — dropping it
flushes pending events and shuts down the transport.

See [`examples/sentry_init.rs`](examples/sentry_init.rs) for a full working
example (`cargo run --features sentry --example sentry_init`).

## Status

Experimental. API is expected to change as backend crates evolve.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE)
at your option.
