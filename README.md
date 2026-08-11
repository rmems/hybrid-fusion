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
- Pure MoE routing math (top-k, normalize, synthetic gate scores) — planned for
  v0.3 / [#26](https://github.com/rmems/hybrid-fusion/issues/26); not tensor matmul
  against checkpoint weights.

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

```rust
use hybrid_fusion::{HybridConfig, HybridNetwork, NeuroModulators, Transformer, SpikingNetwork};
use hybrid_fusion::Tensor;
use hybrid_fusion::Result;

// Implement traits for your backend, then wire them:
// let mut net = HybridNetwork::new(my_transformer, my_snn, HybridConfig::tiny());
// let out = net.forward(&[1u32, 2, 3, 4], None)?;
```

## Public surface

| Item | Purpose |
|------|---------|
| `HybridNetwork<T, S>` | Generic orchestrator over any `Transformer` + `SpikingNetwork`. |
| `Transformer` trait | Backend-agnostic transformer interface. |
| `SpikingNetwork` trait | Backend-agnostic SNN interface. |
| `NeuroModulators` | Neuromodulator struct passed to SNN steps. |
| `HybridConfig` / `TransformerConfig` | Predefined configs (`tiny`, `olmo_1b`). |
| `projector::embed_to_stimuli_with_width` | Pool -> resize -> tanh adapter. |
| `Tensor` | Lightweight owned tensor (data + shape). |

## Guides

- **[Implementing a Backend](docs/implementing-backends.md)** — trait contracts, data flow, tensor shape conventions, and a minimal working example for `Transformer` + `SpikingNetwork`.
- **[Extraction map](docs/extraction-map.md)** — what is extractable from corinth-canal / grok-ozempic into hybrid-fusion vs sibling crates (MoE, dual GGUF+Safetensors, non-extract list).

## Error monitoring (optional)

`hybrid-fusion` ships an optional
[Sentry](https://docs.sentry.io/platforms/rust/) integration. When the
`sentry` feature is enabled:

1. The `sentry` crate is **re-exported** as `hybrid_fusion::sentry` so
   applications share the same crate version and global hub as the library.
2. `HybridNetwork::forward` captures **backend/runtime** failures (e.g. SNN
   step errors) via `telemetry::capture_error`. Caller validation errors
   (`InputLengthMismatch`, config mismatches) are returned without capture
   so routine bad requests do not flood Sentry quota.
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
// Requires: hybrid-fusion = { version = "0.2", features = ["sentry"] }
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
