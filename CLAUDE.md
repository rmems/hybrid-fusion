@AGENTS.md

# CLAUDE.md

Claude-specific notes only. Boundaries, trait contracts, and sibling-crate ownership are in `AGENTS.md`
(imported above); do not duplicate them here. The review checklist is in `REVIEW.md`.

## Commands

Run from the repo root. "Verified" means it ran successfully in a fresh checkout on Rust 1.99.0 (Linux, CPU only).
`rust-toolchain.toml` pins 1.99.0 with rustfmt and clippy. `Cargo.lock` is gitignored at the root, so do not pass `--locked` there.

| Command | Status |
|---|---|
| `cargo build --all-features` | verified |
| `cargo test --all-features` | verified (185 unit tests plus integration tests, 0 failed) |
| `cargo test` | verified |
| `cargo fmt --check` | verified |
| `cargo clippy --all-targets --all-features -- -D warnings` | verified |
| `cargo build --features backends` | verified |
| `cargo test --features "neuromod backends"` | verified |
| `cd smoke/cortex-neuromod && cargo test --locked` | verified (4 passed; fetches `cortex-tensor` by git rev) |
| `cargo test --features sentry` | not run here (`--all-features` already covers it; CI's MSRV job skips it on purpose) |

CI (`.github/workflows/ci.yml`) has two jobs. `validate` runs fmt, clippy, build, and test with `--all-features` on floating
`stable` (`RUSTUP_TOOLCHAIN: stable`), plus the smoke test. `msrv` runs build and test on 1.99.0 for default, `neuromod`,
`backends`, and `neuromod backends`, plus the smoke test. Run the fmt and clippy gates before committing; CI fails on either.

## Architecture map

- `src/traits.rs`: pluggable surface (`Transformer`, `SpikingNetwork`, `ExpertRouter`, `SpikeActivity`, deprecated loader shims).
- `src/hybrid.rs`: `HybridNetwork<T, S>` forward path. `src/projector.rs`: pool, resize, `tanh` into SNN stimuli.
- `src/reverse.rs` + `src/routing.rs`: `ReverseHybridPath` (activity to embedding to MoE route) and pure routing math.
- `src/topology.rs`, `src/plan.rs`, `src/capabilities.rs` + `src/capabilities/contract.rs`: neutral planning topology,
  `HybridExecutionPlan` stage graph, and backend capability negotiation.
- `src/tensor.rs`, `src/validate.rs`, `src/error.rs`, `src/types.rs`, `src/telemetry.rs`: shared tensor type, shape and
  finite-value validation, `HybridError`, config/output types, optional Sentry reporting.
- `src/backends/` (feature `backends`): reference toy backends. `src/neuromod_adapter.rs` (feature `neuromod`): real SNN adapter.
- `tests/`: integration tests, including `tensor_properties.rs` (proptest). `examples/`: some need `sentry`, `backends`, or `neuromod`.
- `smoke/cortex-neuromod/`: separate package and lockfile; the unpublished `cortex-tensor` git pin stays out of the library graph.
- `docs/extraction-map.md` and `docs/implementing-backends.md` are user docs. `docs/superpowers/` is agent planning history.

## Invariants and non-goals (from `REVIEW.md` and `Cargo.toml`)

- No concrete backend types in the public API. The `neuromod` dependency is the one tracked, feature-gated exception (issue #44).
- Do not add a loader abstraction or any checkpoint parsing, mmap, or CUDA code here.
- Stimuli are in `[-1, 1]`; hidden, pooled, and stimuli buffers must be finite; zero-length axes and empty `token_ids` are rejected.
- Shape mismatches return an error, never a silent fallback. `global_step` must not change when forward validation fails.
- No `unsafe` without a safety comment. Add a test for every new path, including edge cases.
- Every `.rs` file starts with `// SPDX-License-Identifier: MIT OR Apache-2.0`.

## Gotchas

- Keep the toolchain pin in sync across `Cargo.toml` `rust-version`, `rust-toolchain.toml`, `smoke/cortex-neuromod/Cargo.toml`,
  `.agents/setup`, and the CI MSRV job.
- The first all-features build compiles Sentry's transitive tree (reqwest, hyper, tokio, ring) and is slow; clippy took about 3.5 minutes cold.
- The smoke test needs network access to fetch `cortex-tensor` from GitHub.
- `Cargo.toml` has an `exclude` list for publishing; `CLAUDE.md` is listed there next to `AGENTS.md`.
- Never run `cargo publish`.
- Edit only what the task needs.
