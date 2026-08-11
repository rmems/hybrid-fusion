# Extraction map — LLM / SNN / MoE contracts into hybrid-fusion

This document maps **research reference** modules into **orchestration
contracts** owned by `hybrid-fusion` versus concrete backends owned by sibling
crates. It is the planning surface for epic
[#20](https://github.com/rmems/hybrid-fusion/issues/20) (v0.3 architecture
contracts). Implementation work lives in the linked sub-issues; this file is
docs-only.

| Source repo | Local path | Role |
|-------------|------------|------|
| [corinth-canal](https://github.com/rmems/corinth-canal) | `~/rmems/corinth-canal` | SNN → embedding → **MoE Router**; GGUF + Safetensors inspect/bridge; pure routing math |
| [grok-ozempic](https://github.com/rmems/grok-ozempic) | `~/rmems/grok-ozempic` | MoE-aware precision tiers, manifest selection, dry-run planner, GOZ1 packing |

**Default rule:** `hybrid-fusion` takes **traits, pure types, and pure math**.
Parsers, mmap I/O, neuron dynamics, and CUDA stay out of this crate.

---

## Status ladder (from corinth-canal)

Promotion stages for research → modular crates
([corinth-canal `docs/PROMOTION_RULES.md`](https://github.com/rmems/corinth-canal/blob/main/docs/PROMOTION_RULES.md)):

1. **reference** — works end-to-end only in the research repo; not portable yet.
2. **stabilizing** — API has stopped thrashing; still tied to unpromoted helpers.
3. **proven** — green against the validation matrix; external surface frozen.
4. **frozen** — copied into the target modular crate; research copy is historical.

When a row below says “extract as trait,” the **first** hybrid-fusion landing is
usually a **stabilizing** orchestration contract (signatures + unit tests), not a
full **frozen** port of the research implementation.

---

## Forward path today (already in hybrid-fusion)

```text
token_ids → Transformer::hidden_states → projector (pool/resize/tanh)
         → SpikingNetwork::step → HybridOutput { embedding, stimuli, fired_neurons, … }
```

| Concern | hybrid-fusion surface | Owner of concrete impl |
|---------|----------------------|------------------------|
| Transformer hidden states | `Transformer` trait | `cortex-tensor` |
| SNN step + channel width | `SpikingNetwork` trait | `neuromod` / `brainstem-daemon` |
| Hidden → stimuli | `projector::embed_to_stimuli_with_width` | this crate (pure) |
| GGUF layout contract | `GgufLoader` / `GgufLayout` | concrete loaders → `engram-parser` |
| Orchestrator | `HybridNetwork<T, S>` | this crate |
| Config / output | `HybridConfig`, `TransformerConfig`, `HybridOutput` | this crate |
| Optional error reporting | `sentry` feature + `telemetry` | optional; apps init DSN |

---

## Target reverse path (v0.3 epic)

```text
SpikeActivity → ProjectionMode → ExpertRouter (MoE)
             → HybridOutput { selected_experts, expert_weights, routing_entropy, … }
```

Checkpoint inventory for MoE onboarding stays **format-agnostic** at the
orchestration layer: GGUF (`GgufLoader`) **and** Safetensors
(`SafetensorsLoader` / layout types, [#27](https://github.com/rmems/hybrid-fusion/issues/27)).

---

## Extraction tables

### A. Projector / spike → embedding

| Source module | Local path | Target hybrid-fusion type/trait | Owner crate | Issue | Notes |
|---------------|------------|----------------------------------|-------------|-------|-------|
| `ProjectionMode` + `Projector` | corinth-canal `src/projector.rs`, `src/types.rs` | `ProjectionMode` enum (`RateSum`, `TemporalHistogram`, `MembraneSnapshot`, `SpikingTernary`) + project hooks | **hybrid-fusion** (contract); learned weights stay backend-side | [#25](https://github.com/rmems/hybrid-fusion/issues/25) | Research status: **stabilizing** |
| Spike activity shapes | corinth-canal projector docs / funnel activity | `SpikeActivity` (or equivalent) pure types | **hybrid-fusion** | [#22](https://github.com/rmems/hybrid-fusion/issues/22) | Indices, rates, membrane snapshots — no GIF dynamics |
| Forward projector (ANN→SNN) | hybrid-fusion `src/projector.rs` | keep existing pool/resize/tanh | **hybrid-fusion** | — | Distinct from reverse-path `ProjectionMode` |

### B. MoE ExpertRouter / routing math

| Source module | Local path | Target hybrid-fusion type/trait | Owner crate | Issue | Notes |
|---------------|------------|----------------------------------|-------------|-------|-------|
| `Router` host | corinth-canal `src/moe/mod.rs` | `ExpertRouter` trait (activity/embedding → expert selection) | **hybrid-fusion** | [#22](https://github.com/rmems/hybrid-fusion/issues/22) | Research: **stabilizing**; no GGUF load in trait |
| Pure gate / top-k math | corinth-canal `src/moe/routing.rs` (`synthetic_gate_scores`, normalize, top-k patterns) | pure helpers (e.g. `src/routing.rs`) + mock router | **hybrid-fusion** | [#26](https://github.com/rmems/hybrid-fusion/issues/26) | Low-risk pure math; unit-test without checkpoints |
| Checkpoint gate matmul | corinth-canal `routing.rs` `checkpoint_gate_scores` / `safetensors_gate_scores` | **do not extract** as hybrid-fusion I/O | `engram-parser` + research | — | Needs real weights / mmap |
| Family adapters | corinth-canal `src/moe/adapter.rs` | stay research / parser | research / `engram-parser` | — | Olmoe, Qwen3Moe, Gemma4, … tensor names |
| MoE fields on output | corinth-canal `ModelOutput`-style fields | expand `HybridOutput` (`expert_weights`, `selected_experts`, optional `routing_entropy`) | **hybrid-fusion** | [#22](https://github.com/rmems/hybrid-fusion/issues/22) | Keep existing ANN→SNN fields working |
| Reverse-path integration test | mock backends pattern in hybrid-fusion `backends` | activity → project → route tests | **hybrid-fusion** | [#23](https://github.com/rmems/hybrid-fusion/issues/23) | Depends on #22, #25, #26 |

### C. Dual checkpoint contracts (GGUF + Safetensors)

Orchestration must stay **format-agnostic**: MoE candidate discovery and dry-run
inventory talk to layout types, not payload parsers.

| Source module | Local path | Target hybrid-fusion type/trait | Owner crate | Issue | Notes |
|---------------|------------|----------------------------------|-------------|-------|-------|
| GGUF layout façade | hybrid-fusion `GgufLoader` / `GgufLayout`; corinth-canal `src/moe/gguf/` | keep/extend layout trait; no mmap here | **hybrid-fusion** (trait); **engram-parser** (parse/mmap/dequant) | existing | Concrete GGUF stays out of this crate |
| Safetensors header inspect / manifest | corinth-canal `src/moe/safetensors.rs`, `moe/safetensors/discovery.rs` | `SafetensorsLoader` and/or unified `CheckpointLoader` + layout/manifest types (name, dtype, shape, shard refs, labels) | **hybrid-fusion** (contract); **engram-parser** (I/O) | [#27](https://github.com/rmems/hybrid-fusion/issues/27) | Header-only inspect in research; dual format for MoE onboarding |
| MoE router/expert candidates | corinth-canal safetensors discovery | candidate / router-tensor **role** types for #22 / #24 / #26 | **hybrid-fusion** | [#27](https://github.com/rmems/hybrid-fusion/issues/27) | Name + shape heuristics only at trait layer |
| GGUF checkpoint façade | corinth-canal `src/moe/checkpoint.rs`, `gguf/*` | not promoted into hybrid-fusion body | **engram-parser** | — | Research status: **reference** |

### D. Precision tiers + dry-run planner (grok-ozempic)

| Source module | Local path | Target hybrid-fusion type/trait | Owner crate | Issue | Notes |
|---------------|------------|----------------------------------|-------------|-------|-------|
| Precision policy | grok-ozempic `src/core/precision.rs` | precision-tier types (`preserve` / `fp16` / `ternary_snn` concepts) | **hybrid-fusion** (types) | [#24](https://github.com/rmems/hybrid-fusion/issues/24) | No GOZ1 packing |
| Tensor classification | grok-ozempic `src/core/selection.rs` | classification hooks / enums usable by planner | **hybrid-fusion** (optional thin types) | [#24](https://github.com/rmems/hybrid-fusion/issues/24) | Manifest-driven logic can stay research initially |
| Dry-run planner | grok-ozempic `src/core/dry_run.rs` | `HybridStagePlanner` (or equivalent) dry-run report types | **hybrid-fusion** | [#24](https://github.com/rmems/hybrid-fusion/issues/24) | Plan kernel calls without loading full weights |
| `BackendKernel` trait | grok-ozempic `src/core/backend.rs` | **not** hybrid-fusion’s primary surface | `myelin-accelerator` / research | — | Kernel ownership is outside orchestrator |
| GOZ1 / weight pack | grok-ozempic `weight_pack*`, `artifact.rs` | **non-extract** | **grok-ozempic** | — | Grok-specific container |

### E. Spike activity / funnel / dynamics (mostly non-extract)

| Source module | Local path | Target | Owner crate | Notes |
|---------------|------------|--------|-------------|-------|
| GIF / funnel hidden layer | corinth-canal `src/funnel.rs` | dynamics only | **neuromod** / research | **reference**; not hybrid-fusion |
| LIF / GIF CUDA kernels | corinth-canal `src/gpu/*` | kernels | **myelin-accelerator** / research | CUDA out of hybrid-fusion |
| Telemetry encoder | corinth-canal `src/telemetry.rs` | sensory / telemetry | **axon-encoder** / research | Optional later; not this epic |
| SNN runtime scheduling | — | — | **brainstem-daemon** | sibling ownership |
| Neuromodulator critic | hybrid-fusion `NeuroModulators` already | mapping / critic | **limbic-critic** | hybrid-fusion keeps the struct passed into `step` |

---

## Explicit non-extract list

Do **not** copy these into `hybrid-fusion` as implementation work:

| Item | Why | Where it stays |
|------|-----|----------------|
| **SAAQ / latent calibration** | Still research; dual-SAAQ validation gates | corinth-canal `src/latent.rs`, experiments |
| GGUF / Safetensors **mmap, parse, dequant** | Crate boundary: I/O parsers | `engram-parser` + research `moe/gguf`, `moe/safetensors` payload paths |
| GIF / LIF / funnel neuron dynamics | Dynamics ownership | `neuromod`, corinth-canal `funnel` |
| CUDA kernels / fatbins | Accelerator ownership | `myelin-accelerator`, corinth-canal `gpu/` |
| GOZ1 packing / real Grok-1 quant pipeline | Research product surface | `grok-ozempic` |
| Machine-local paths (`$HOME/Downloads/...`, lineup TOMLs with host paths) | Non-portable | research configs only |
| Family-specific tensor name tables (Olmoe vs Qwen3Moe vs Grok, …) | Parser / research adapters | `engram-parser` / corinth-canal `adapter.rs` |
| Cloud provider provisioning | Infra | Dioscuri-Cloud / external |

---

## Sibling crate ownership (quick matrix)

| Crate | Responsibility relative to this map |
|-------|-------------------------------------|
| **hybrid-fusion** | Orchestration traits/types; pure projector + pure MoE math; dual checkpoint **contracts** |
| **cortex-tensor** | Transformer / tensor math backends implementing `Transformer` |
| **engram-parser** | Concrete GGUF + Safetensors loaders implementing layout traits |
| **neuromod** | Neuron dynamics implementing `SpikingNetwork` |
| **brainstem-daemon** | SNN runtime scheduling |
| **limbic-critic** | Neuromodulator → critic signal mapping |
| **axon-encoder** | Sensory / telemetry encoding |
| **myelin-accelerator** | Reusable CUDA kernels (ternary GEMM, SAAQ reductions, etc.) |
| **corinth-canal** | Reference E2E research (not a production dependency of hybrid-fusion) |
| **grok-ozempic** | Grok-1 quant / dry-run research (not a production dependency) |

---

## Sub-issue DAG (implementation order)

```text
#20 epic: extract LLM/SNN architecture contracts
 │
 ├─ 1. #21  docs: this extraction map          ← docs-only (this file)
 ├─ 2. #22  ExpertRouter + SpikeActivity traits
 ├─ 3. #25  ProjectionMode contract
 ├─ 4. #26  pure MoE router math
 ├─ 5. #24  precision-tier + HybridStagePlanner
 ├─ 6. #23  mock reverse-path tests
 └─ 7. #27  SafetensorsLoader / layout types
```

#9 (optional Sentry) is **outside** this epic and is already implemented in-tree.

---

## Acceptance checklist (issue #21)

- [x] Map covers projector, MoE ExpertRouter / routing math, precision tiers, dry-run planner, SpikeActivity types
- [x] Map covers `moe/safetensors` → hybrid-fusion SafetensorsLoader/layout traits (#27); concrete ST/GGUF I/O → `engram-parser`
- [x] Dual checkpoint row: existing `GgufLoader` + planned Safetensors; MoE path format-agnostic at orchestration layer
- [x] Explicit non-extract list: SAAQ, GGUF/ST mmap parse, GIF dynamics, CUDA, GOZ1, machine-local paths
- [x] Links to local source paths + sibling crate owners
- [x] Status ladder note (reference → stabilizing → proven → frozen)

---

## Related docs

- [Implementing a Backend](implementing-backends.md) — how to implement `Transformer` / `SpikingNetwork` / `GgufLoader` today
- [AGENTS.md](../AGENTS.md) — crate boundaries for coding agents
- Epic: [#20](https://github.com/rmems/hybrid-fusion/issues/20)
- Milestone: [v0.3 — LLM/SNN architecture contracts](https://github.com/rmems/hybrid-fusion/milestone/2)
