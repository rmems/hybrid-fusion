# Design: Reverse-path orchestrator (`ReverseHybridPath`) — issue #23

**Status:** implemented (see plan `docs/superpowers/plans/2026-08-15-reverse-path-orchestrator.md`)  
**Date:** 2026-08-14  
**Issue:** [rmems/hybrid-fusion#23](https://github.com/rmems/hybrid-fusion/issues/23)  
**Epic:** [#20](https://github.com/rmems/hybrid-fusion/issues/20)  
**Approach:** **B** — host type extracted from corinth-canal `Model` reverse half  

---

## 1. Problem

hybrid-fusion already owns reverse-path **contracts and pure math**:

| Piece | Status |
|-------|--------|
| `SpikeActivity`, `ExpertRouter`, `ExpertRouteOutput` | #22 |
| `ProjectionMode`, `project_spike_activity` | #25 |
| `routing` + `SyntheticExpertRouter` / `StubExpertRouter` | #26 |

What is missing is the **orchestrator** corinth-canal uses to close the loop:

```text
// corinth-canal src/model/core.rs — Model::forward_activity
spike_train + potentials + iz
  → projector.project(...) → embedding
  → router.forward(&embedding) → expert_weights, selected_experts
  → ModelOutput { embedding, expert_weights: Some(...), selected_experts: Some(...), ... }
```

Names above are the **corinth-canal** source API. In the hybrid-fusion contract
they map to `project_spike_activity(...)` and `ExpertRouter::route(...)`; there
is no `forward` method on `ExpertRouter`.

Issue #23 acceptance is an integration path with MoE fields asserted, without checkpoints or SAAQ. Extraction goal: hybrid-fusion owns this orchestration so corinth-canal can become thinner.

---

## 2. Goals

1. Public reverse-path host mirroring corinth `Model`’s **projector + router** half (not telemetry/GIF/GPU).
2. Integration tests: activity → project → route → `HybridOutput` MoE fields.
3. File-free only (`SyntheticExpertRouter` / `StubExpertRouter`).
4. Document how real MoE backends plug in later (`ExpertRouter` + engram-parser).
5. Keep ANN→SNN `HybridNetwork::forward` unchanged (MoE fields remain `None`).

## 3. Non-goals

- GGUF / Safetensors load or gate matmul
- GIF / learned projector W/b (stays neuromod / research)
- SAAQ / latent
- CUDA / GPU temporal
- `#24` dry-run planner (optional smoke only if already present — **skip**)
- Changing `HybridNetwork` generics to include `ExpertRouter`

---

## 4. Architecture

Two parallel hosts (do not merge into one mega-generic):

```text
ANN → SNN (existing)
  HybridNetwork<T: Transformer, S: SpikingNetwork>
    token_ids → hidden → stimuli → snn.step → HybridOutput (MoE = None)

SNN activity → MoE (new — extract from corinth Model)
  ReverseHybridPath<R: ExpertRouter>
    SpikeActivity → project_spike_activity → router.route → HybridOutput (MoE = Some)
```

### Data flow

```text
SpikeActivity { spike_train, potentials, iz_potentials }
        │
        ▼  project_spike_activity(mode, activity, n_neurons, embed_dim)
embedding: Vec<f32>   (len == embed_dim, tanh-bounded)
        │
        ▼  router.route(&embedding)
ExpertRouteOutput { expert_weights, selected_experts, routing_entropy }
        │
        ▼
HybridOutput {
  embedding,
  stimuli: [],                    // reverse path has no ANN→SNN stimuli
  fired_neurons: last spike step, // telemetry from activity; not SNN step output
  global_step,
  expert_weights: Some(...),
  selected_experts: Some(...),
  routing_entropy: ...
}
```

`expert_weights` is **dense** — one entry per expert (`len == num_experts`,
sum ≈ 1). `selected_experts` is **sparse** — only the top-k indices
(`len == top_k`, each `< num_experts`). They are not aligned element-wise;
index into `expert_weights` with a value from `selected_experts`.

---

## 5. Public API

### 5.1 Type

```rust
/// Reverse-path host: SNN activity → embedding → MoE route.
///
/// Extracted from corinth-canal `Model` (`projector` + `router` half of
/// `forward_activity`). Does not own neuron dynamics or checkpoint I/O.
pub struct ReverseHybridPath<R: ExpertRouter> {
    mode: ProjectionMode,
    n_neurons: usize,
    embed_dim: usize,
    router: R,
    global_step: u64,
}
```

Fields are private; expose accessors `mode()`, `n_neurons()`, `embed_dim()`, `router()` / `router_mut()`, `global_step()`, `reset()`.

### 5.2 Construction

```rust
impl<R: ExpertRouter> ReverseHybridPath<R> {
    /// Rejects `n_neurons == 0` or `embed_dim == 0`.
    pub fn new(
        mode: ProjectionMode,
        n_neurons: usize,
        embed_dim: usize,
        router: R,
    ) -> Result<Self>;
}
```

Does **not** validate `router.num_experts()` / `top_k()` beyond what the router already enforces at its `new`.

### 5.3 `forward_activity`

```rust
pub fn forward_activity(&mut self, activity: &SpikeActivity) -> Result<HybridOutput>
```

Semantics (match corinth `Model::forward_activity`):

1. `embedding = project_spike_activity(self.mode, activity, self.n_neurons, self.embed_dim)?`
2. `route = self.router.route(&embedding)?`
3. `global_step = global_step.saturating_add(1)`  
   - Optional: capture router errors with `telemetry::capture_error` for consistency with SNN step failures on the forward path (only backend/runtime errors). Prefer capture on `route` errors that are not pure validation; simplest rule: **do not Sentry pure validation; capture other `HybridError` variants if already used that way** — for v1, **propagate without Sentry** unless a clear backend error variant exists. Keep parity with current forward path: capture only when analogous to SNN step. Router errors can be `InvalidConfig` or success-path failures; **v1: no Sentry capture** on reverse path (document; avoids flooding on bad activity).
4. Build `HybridOutput`:
   - `embedding` from step 1
   - `stimuli: Vec::new()`
   - `fired_neurons`: last non-empty step of `activity.spike_train`, else empty  
     (indices already validated by project for this `n_neurons`)
   - `global_step` from step 3
   - MoE fields from `route`

### 5.4 Module layout

| File | Change |
|------|--------|
| `src/reverse.rs` (new) | `ReverseHybridPath` |
| `src/lib.rs` | `pub mod reverse`; re-export `ReverseHybridPath` |
| `tests/reverse_path.rs` | integration tests (`required-features` / only with `backends` for routers) |
| `docs/implementing-backends.md` | reverse orchestrator section |
| `README.md` / `CHANGELOG.md` | surface + unreleased note |
| `AGENTS.md` | one line: reverse host vs ANN→SNN host |

Integration tests that need `SyntheticExpertRouter` require `--features backends` (or `all-features`). Prefer `[[test]]` with `required-features = ["backends"]` if the crate supports it; else gate tests with `#[cfg(feature = "backends")]` and document `cargo test --features backends`.

**Preferred:** single integration file `tests/reverse_path.rs` that always compiles: use pure functions + a **local mock** `ExpertRouter` in the test file (no backends feature required). Also one test using `SyntheticExpertRouter` under `#[cfg(feature = "backends")]`. That maximizes CI coverage without feature pitfalls.

---

## 6. Testing plan

### 6.1 Unit tests (`src/reverse.rs`)

- `new` rejects zero `n_neurons` / `embed_dim`
- `forward_activity` increments `global_step`
- `reset` clears step

Use a tiny in-module mock router (or `StubExpertRouter` with cfg).

### 6.2 Integration (`tests/reverse_path.rs`)

| Test | Assert |
|------|--------|
| Happy path (mock or Synthetic) | `selected_experts.len() == top_k`, weights sum ≈ 1, `embedding.len() == embed_dim`, MoE fields `Some` |
| Second `ProjectionMode` | still routes (e.g. `MembraneSnapshot`) |
| Determinism | same activity twice → same weights + selection |
| ANN→SNN still clean | existing hybrid tests unchanged |

### 6.3 Constraints

- No filesystem, network, or machine-local paths
- No real checkpoints

---

## 7. Documentation for real backends

In implementing-backends reverse section:

1. Implement `ExpertRouter` with checkpoint-backed gate matmul (engram-parser + cortex-tensor later).
2. Construct `ReverseHybridPath::new(mode, n_neurons, embed_dim, my_router)`.
3. Call `forward_activity` with live `SpikeActivity` from neuromod / funnel.
4. Do **not** use this path for SAAQ.

---

## 8. Error handling

| Condition | Error |
|-----------|--------|
| `n_neurons == 0` or `embed_dim == 0` at `new` | `InvalidConfig` |
| Bad activity (potentials length, OOB spikes, non-finite) | from `project_spike_activity` |
| Empty embedding / router validation | from `router.route` |

---

## 9. Naming

- Type: **`ReverseHybridPath`** (clear, not “Model” — avoids clash with research `Model`)
- Method: **`forward_activity`** (match corinth name for extractability)

---

## 10. Implementation order

1. `src/reverse.rs` + unit tests  
2. Public re-exports  
3. `tests/reverse_path.rs`  
4. Docs + CHANGELOG  
5. `cargo test --all-features` + clippy  

---

## 11. Acceptance mapping (#23)

| Criterion | Design response |
|-----------|-----------------|
| Integration test asserts MoE fields | `tests/reverse_path.rs` top-k + weight sum |
| ≥1 ProjectionMode + mock ExpertRouter | RateSum + mock/Synthetic |
| No machine-local paths / network | pure only |
| Document extend with real MoE backends | implementing-backends section |

---

## 12. Spec self-review

| Check | Result |
|-------|--------|
| Placeholders | None material |
| Consistency | Dual-host split; MoE fields on reverse only |
| Scope | Single PR; no #24/#27 |
| Ambiguity | `stimuli` empty; `fired_neurons` = last spike step; no Sentry on reverse v1 |

---

## 13. Open for implementer (resolved defaults)

| Topic | Default in this spec |
|-------|----------------------|
| Type name | `ReverseHybridPath` |
| ANN fields on reverse output | empty `stimuli`; last-step `fired_neurons` |
| Sentry | none on reverse v1 |
| backends feature | mock router in tests without feature; optional Synthetic under cfg |
