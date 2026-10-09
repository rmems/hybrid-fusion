# Implementing a Backend

`hybrid-fusion` is backend-agnostic: it defines the `Transformer`, `SpikingNetwork`,
`GgufLoader`, `SafetensorsLoader`, `ExpertRouter`, `SpikeActivity`, and
`NeuroModulators` contracts but ships no concrete math. This guide explains every
trait method, how data flows through `HybridNetwork::try_new` /
`HybridNetwork::forward`, and provides a
minimal compilable example you can adapt for your own backend.

For authoritative signatures, always refer to [`src/traits.rs`](../src/traits.rs).

---

## Table of Contents

1. [Transformer trait](#1-transformer-trait)
2. [SpikingNetwork trait](#2-spikingnetwork-trait)
3. [NeuroModulators](#3-neuromodulators)
4. [GgufLoader / GgufLayout](#4-ggufloader--gguflayout)
5. [SafetensorsLoader / SafetensorsLayout](#5-safetensorsloader--safetensorslayout)
6. [How construction and the forward pipeline work](#6-how-construction-and-the-forward-pipeline-work)
7. [Tensor shape conventions](#7-tensor-shape-conventions)
8. [Minimal working example](#8-minimal-working-example)
9. [Common pitfalls](#9-common-pitfalls)
10. [Error reference](#10-error-reference)
11. [Reverse path: SpikeActivity + ExpertRouter + ReverseHybridPath](#11-reverse-path-spikeactivity--expertrouter--reversehybridpath)

---

## 1. Transformer trait

```rust
pub trait Transformer {
    fn hidden_states(&self, token_ids: &[u32]) -> Tensor;
    fn dim(&self) -> usize;
    fn max_seq_len(&self) -> usize;
    fn param_count(&self) -> usize;
}
```

### `hidden_states(&self, token_ids: &[u32]) -> Tensor`

Returns the model's hidden-state representation for the given token sequence.

**Contract:**

- `token_ids` is guaranteed non-empty and `<= max_seq_len()` by the time
  `HybridNetwork::forward` calls this method. You do not need to validate
  length yourself, but defensive checks are fine.
- The returned `Tensor` **must** have one of these shapes. Every other rank
  (0, 3, …) is rejected with `HybridError::HiddenStateRank` **before**
  pooling or `SpikingNetwork::step`:
  - **2-D** `[seq_len, dim]` — canonical per-token hidden states.
    `shape[0]` **must equal** `token_ids.len()` (`HiddenStateSeqLen`) and
    `shape[1]` **must equal** `dim()` (`HiddenStateDim`).
  - **1-D** `[dim]` — an explicitly supported pre-pooled embedding. The
    single axis **must equal** `dim()`. Sequence length is not encoded.
- Backing `data.len()` must equal the layout product (`seq_len * dim` or
  `dim`); mismatches return `HybridError::HiddenStateDataLen`.
- Every value must be finite (no NaN or ±Inf); otherwise
  `HybridError::NonFinite { stage: HiddenState, … }`.
- `dim()` must be `> 0`. A zero hidden dimension is rejected as
  `HiddenStateDim { expected: 1, got: 0 }` before pooling or `step`, even
  if a deserialized tensor uses a matching zero-extent axis.
- All shape dimensions must be `> 0`. `Tensor::from_vec` panics on zero-dim
  shapes.

**Common implementation pattern:**

```rust
fn hidden_states(&self, token_ids: &[u32]) -> Tensor {
    let seq_len = token_ids.len();
    // ... run your transformer forward pass ...
    let data: Vec<f32> = /* fill with seq_len * dim values */;
    Tensor::from_vec(data, &[seq_len, self.dim])
}
```

### `dim(&self) -> usize`

The hidden-state embedding dimension. This must be consistent with the
tensor shapes returned by `hidden_states`. The pipeline validates
`shape[1] == dim()` for rank 2 and `shape[0] == dim()` for rank 1.

### `max_seq_len(&self) -> usize`

Maximum sequence length the transformer accepts. `HybridNetwork::forward`
rejects inputs exceeding this with `HybridError::InputLengthMismatch`.

### `param_count(&self) -> usize`

Total number of parameters. This is purely informational — the pipeline
never calls it during inference. Return `0` if you don't track parameters.

---

## 2. SpikingNetwork trait

```rust
pub trait SpikingNetwork {
    fn step(&mut self, stimuli: &[f32], modulators: &NeuroModulators) -> Result<Vec<usize>>;
    fn num_channels(&self) -> usize;
    fn num_neurons(&self) -> usize { self.num_channels() }
}
```

### `step(&mut self, stimuli: &[f32], modulators: &NeuroModulators) -> Result<Vec<usize>>`

Advance the SNN by one timestep.

**Contract:**

- `stimuli.len()` equals `num_channels()`. The projector produces exactly
  this many values by resizing the transformer's hidden-state embedding.
- All values in `stimuli` are in `[-1.0, 1.0]` — they have been squashed
  through `tanh` by the projector.
- `modulators` is always provided (defaults are used if the caller passes
  `None` to `HybridNetwork::forward`).
- Return a `Vec<usize>` of neuron indices that fired this step. Indices
  must be `< num_neurons()`. An empty `Vec` means no neurons
  fired.
- Return `Err(HybridError::SnnStep(...))` if the step encounters an
  internal error.

**Common implementation pattern:**

```rust
fn step(&mut self, stimuli: &[f32], modulators: &NeuroModulators) -> Result<Vec<usize>> {
    let mut fired = Vec::new();
    for (i, &input) in stimuli.iter().enumerate() {
        // Apply neuromodulator scaling, update neuron state, check threshold
        let effective = input * modulators.dopamine;
        // ... neuron dynamics ...
        if /* threshold crossed */ {
            fired.push(i);
        }
    }
    Ok(fired)
}
```

### `num_channels(&self) -> usize`

The number of input channels the SNN expects. This determines the length
of the `stimuli` slice passed to `step`. The projector resizes the
transformer embedding to match this width.

**Zero channels are rejected.** `HybridNetwork::forward` returns
`HybridError::ZeroSnnChannels` before projection or `step` when
`num_channels()` is `0`.

**Note:** `num_channels` is independent from `Transformer::dim()`. The
transformer might produce a 128-dim embedding while the SNN only has 64
input channels — the projector handles the resize.

### `num_neurons(&self) -> usize`

The nonzero output population. Fired IDs are in `0..num_neurons()`, independent
of input width. Both dimensions must remain stable over stepping. The default
returns `num_channels()` for one-output-per-input backends; unequal-width
implementations **must override it**. `HybridNetwork` rejects a zero output
population before stepping and out-of-domain IDs afterward. A failed output
check leaves the host counter unchanged but cannot roll back the backend step.

### The real SNN backend: `NeuromodSnn` (optional `neuromod` feature)

`hybrid-fusion` ships one production-oriented `SpikingNetwork` implementation:
`NeuromodSnn`, behind the optional `neuromod` feature. It adapts
[`neuromod`](https://crates.io/crates/neuromod) 0.7, a genuine spiking engine
with LIF and Izhikevich neuron dynamics, to this crate's `SpikingNetwork`
contract.

```sh
cargo build --features neuromod
```

`SimpleSnn` (behind the `backends` feature) is a deterministic reference/mock
only. It exists for tests and examples and does not model neuron dynamics.
When you need real dynamics, enable `neuromod` and use `NeuromodSnn`.

`NeuromodSnn` keeps every `neuromod` type out of the public API. Callers work
only with hybrid-fusion vocabulary: `NeuroModulators` in, `Result<Vec<usize>>`
out, and `HybridError` on failure. The modulator conversion (see
[section 3](#3-neuromodulators)) and the mapping of the engine's step errors to
`HybridError` happen internally.

**Seeded replay.** `NeuromodSnn` owns its random generator, so stepping is
deterministic by construction. Build with `NeuromodSnn::with_seed(num_channels,
num_neurons, seed)?` and two instances given the same dimensions, seed and stimulus sequence
emit identical fired-index sequences (the trait `step` routes through the owned
generator, backed by neuromod's `step_with_rng`). `NeuromodSnn::new(num_channels, num_neurons)?`
uses a fixed default seed. `reset()` resets the neuron dynamics and restores
initial weights and thresholds, but does not
reseed the generator; call `reseed(seed)` to restart the random stream
explicitly. Reset plus reseed replays a run from a known starting point.

**Width and memory.** `new` / `with_seed` require explicit input and output
dimensions. All LIF outputs retain full-width input connectivity. The internal
Izhikevich neuron does not contribute returned IDs. Weights, eligibility, and
the replay weight snapshot use 16 × inputs × outputs bytes in neuromod 0.7:
4-byte weights + 8-byte traces + 4-byte snapshot entries. Constructors reject
zero dimensions, overflow, and matrix payloads above `MAX_MATRIX_BYTES` (64 MiB)
before allocating. This fixed guard allows up to 256 outputs at 16,384 inputs
without permitting the former 4 GiB matrix construction. It is a conservative
adapter allocation policy, not a neuron-dynamics limit or a total-memory bound.

### RM-1940 migration

This is a source and serialized-output compatibility change. There is no
implicit default output population and no `MAX_LIF_NEURONS` cap:

```rust
use hybrid_fusion::{NeuromodSnn, NeuroModulators, SpikeActivity, SpikingNetwork};

// Explicitly opt into PR #49's reduced topology (not per-input output semantics).
let mut snn = NeuromodSnn::with_seed(16_384, 8, 42)?;
let fired = snn.step(&vec![0.5; snn.num_channels()], &NeuroModulators::default())?;
let activity = SpikeActivity::from_fired(&fired, snn.num_neurons())?;
assert_eq!(activity.potentials.len(), 8);
// ReverseHybridPath::new(mode, snn.num_neurons(), embed_dim, router)?
# Ok::<(), hybrid_fusion::HybridError>(())
```

- Replace `new(inputs)` with `new(inputs, outputs)?`, and
  `with_seed(inputs, seed)` with `with_seed(inputs, outputs, seed)?` (or handle
  the returned `Result` explicitly). Choose the output population before
  construction; allocation rejection is not an instruction to silently retry
  with fewer neurons.
- To retain a pre-#49 topology at a modest width, explicitly choose equal
  input/output counts. Choosing `(width, width.min(8))` retains #49's topology.
  Populations greater than eight are supported; no spikes are duplicated.
- Replace `from_fired(fired, snn.num_channels())` with
  `from_fired(fired, snn.num_neurons())`. After a host forward, use
  `from_fired(&out.fired_neurons, out.num_neurons)`. Configure the reverse host
  with the same output count, including for silent steps.
- RateSum and SpikingTernary have one feature per output neuron;
  TemporalHistogram has four neuron-major bins per output. MembraneSnapshot
  also has one feature per output, but `from_fired` supplies zero potentials,
  not measured membrane values. `embed_dim` remains an independent resize:
  increasing it does not restore missing neurons or recover old semantics.
- `HybridOutput.num_neurons` is required on struct literals and deserialization.
  Migrate stored records using their actual topology, not `stimuli.len()` or
  `max(fired)+1`. A reduced population has no output positions 8 onward when
  its count is eight; callers must not silently pad activity to input width.
- `HybridConfig.snn_lif_neurons` remains a construction hint, not a second
  runtime source of truth. `HybridNetwork::execution_plan` reads the backend
  population; config-only plans use the hint and reject zero.
- Constructors now return `InvalidConfig` for zero dimensions in both debug
  and release, replacing the old debug assertion / invalid release backend.
  Step input-length, non-finite input/modulator checks and error mapping stay
  unchanged. Reset now also restores retuned thresholds; post-reset output can
  differ from the old adapter's incomplete reset, and agrees with a fresh run.

**Older serialized outputs.** Deserializing a record without `num_neurons`
fails with ``missing field `num_neurons` ``. There is no serde default or
automatic legacy conversion. A caller-owned migration must recover the
actual output count from the producing backend's checkpoint or verified run
metadata, add it to the record, and validate the fired IDs against that count
with `SpikeActivity::from_fired` before reverse projection. This also applies
to reverse-produced outputs, whose `stimuli` is empty. Neither input width,
embedding length nor the largest observed fired ID identifies the population;
silent neurons and silent steps still have an output dimension. If that
provenance is unavailable, keep the record in its legacy format or reject it
for reverse processing until the topology is recovered; do not invent a count.
For known PR #49 `NeuromodSnn` runs only, the actual population was
`min(input_width, 8)`; earlier equal-width runs and custom backends require
their own verified topology, not that blanket conversion.

**Remaining upstream dependency.** `neuromod` 0.7's
[`with_dimensions`](https://github.com/rmems/neuromod/blob/v0.7.0/src/engine.rs#L404-L430)
stores dense weights and eligibility for every neuron/input pair. Reusing
standalone `LifNeuron` primitives with diagonal connectivity would change
topology, predictive currents, inhibition and plasticity orchestration; this
adapter does not implement another engine. Preserving a fully connected
16,384-output plastic network with bounded memory needs an upstream storage /
plasticity design, not an ID remapping here. Sparse connectivity alone would
also change semantics. No full-versus-reduced population or model-quality
equivalence is claimed, and no downstream experiments have been validated.

**Consumer inventory (RM-1940).** `HybridNetwork` preflight, forward projector,
config validation and forward-plan input ports consume **input width**.
`HybridOutput`, forward-plan output metadata, `SpikeActivity::from_fired`,
reverse projectors and `ReverseHybridPath` consume **output population**.
`SimpleSnn`, inline example backends and existing generic mocks have one output
per input, so the trait default preserves their behavior. The neuromod parity
test uses explicit equal dimensions. Historical design notes under
`docs/superpowers` describe old proposals, not the current runtime contract.

The 2026-10-02 downstream source/manifests inventory found no dependencies on
hybrid-fusion or consumers of these APIs at the following main-branch heads:

- [`Limen-Neural/brainstem-daemon`](https://github.com/Limen-Neural/brainstem-daemon/blob/8fe78be5091531c8867a1b8c012a2e52a92b83b8/Cargo.toml#L32-L44)
  is accessible at its canonical path (the earlier `rmems` path returned 404).
  It uses neuromod 0.6 directly and independently configures LIF count,
  Izhikevich count and input channels; it does not consume `HybridOutput`.
- [`rmems/cortex-tensor`](https://github.com/rmems/cortex-tensor/blob/a7de7137695683bb2b3f33750b7f9a667870786b/src/snn/neuromod_adapter.rs#L27-L56)
  has its own `NeuromodNetwork` / `SnnBackend` contract with an equality check
  between total neuron-bank count and channels. It is not this adapter; any
  future bridge needs a separate review of the actual fired-ID domain.
- [`Limen-Neural/neuromod`](https://github.com/Limen-Neural/neuromod/tree/a897cc91f7e97efb465d41247fa907e2c1f2332d)
  is the upstream engine, not a consumer of the changed hybrid-fusion API.
- [`rmems/corinth-canal`](https://github.com/rmems/corinth-canal/blob/06cefa04fe117ab7e2db4c40946d6664aed8440c/src/projector.rs#L204-L240)
  has an independent reverse projector. A future bridge must use the actual
  output population to avoid its filtering of out-of-domain IDs.

No direct API migration was identified in those revisions. This is **source
inspection only**: no downstream repositories were modified, built or run, and
unpublished consumers / experiments remain unverified. A future consumer of
`NeuromodSnn` must use its reported LIF output count, not add its internal
Izhikevich bank. External consumers must still follow the migration above.

**Allocation evidence.** At `(16_384, 8)`, matrix payload is exactly 2,097,152
bytes (512 KiB weights + 1 MiB traces + 512 KiB replay weights), independently
checked against live vectors by the large-width unit test. Additional storage
includes 192 KiB of input timestamps/predictive state, neuron structs, the RNG,
32 bytes of saved thresholds, vector headers and allocator overhead. The old
shape was 3 GiB only if all three matrices were counted as `f32`; actual 8-byte
traces make its payload 4 GiB. Never interpret the matrix estimate as RSS.
To measure peak **whole-process** resident memory separately on Linux:

```sh
cargo build --release --features neuromod --example neuromod_memory
/usr/bin/time -v target/release/examples/neuromod_memory
```

This measures one constructor, no stepping, and includes the executable,
runtime and allocator. RSS is platform-dependent; the PR records observed
values and the exact tested commit rather than making them a portable bound.

---

## 3. NeuroModulators

```rust
pub struct NeuroModulators {
    pub dopamine: f32,       // reward / motivation signal
    pub cortisol: f32,       // stress / threat signal
    pub acetylcholine: f32,  // attention / novelty signal
    pub tempo: f32,          // speed / urgency multiplier
    pub aux_dopamine: f32,   // secondary reward channel
}
```

Default values (via `Default`):

| Field           | Default | Typical range |
|-----------------|---------|---------------|
| `dopamine`      | `0.5`   | `0.0` — `1.0` |
| `cortisol`      | `0.0`   | `0.0` — `1.0` |
| `acetylcholine` | `0.5`   | `0.0` — `1.0` |
| `tempo`         | `1.0`   | `0.5` — `2.0` |
| `aux_dopamine`  | `0.0`   | `0.0` — `1.0` |

### What each field means

- **`dopamine`** — Primary reward modulation. Higher values increase
  excitability and firing likelihood. Derive from reward signals,
  confidence scores, or loss-based feedback.
- **`cortisol`** — Stress / threat signal. Can suppress firing or shift
  neuron dynamics toward conservative behavior. Derive from error rates,
  uncertainty, or anomaly detection.
- **`acetylcholine`** — Attention and novelty gating. Influences how
  strongly new stimuli affect neuron states. Derive from attention weights,
  novelty scores, or input entropy.
- **`tempo`** — Speed multiplier. Values > 1.0 accelerate dynamics
  (urgency), < 1.0 slow them (deliberation). Derive from latency budgets,
  real-time constraints, or task pacing.
- **`aux_dopamine`** — Secondary reward channel for multi-objective setups.
  Same semantics as `dopamine` but can carry an independent signal.

### How to derive from telemetry

```rust
let modulators = NeuroModulators {
    dopamine: reward_signal.clamp(0.0, 1.0),
    cortisol: (error_rate * 2.0).clamp(0.0, 1.0),
    acetylcholine: attention_weight.clamp(0.0, 1.0),
    tempo: if latency_budget_ms < 10 { 1.5 } else { 1.0 },
    aux_dopamine: secondary_reward.clamp(0.0, 1.0),
};
```

If you don't have a specific signal, use the defaults — they represent a
neutral, balanced state.

### Mapping to `NeuromodSnn` (the `neuromod` backend)

`neuromod`'s own modulator vocabulary differs from hybrid-fusion's. It exposes
`{ dopamine, serotonin, acetylcholine, norepinephrine }`, all defaulting to
`0.0`, whereas hybrid-fusion defaults `dopamine` and `acetylcholine` to `0.5`
and `tempo` to `1.0`. `NeuromodSnn` converts explicitly, setting every neuromod
field (no implicit field reuse):

| hybrid-fusion field | neuromod field   | note |
|---------------------|------------------|------|
| `dopamine`          | `dopamine`       | `(dopamine + aux_dopamine).clamp(0.0, 1.0)` |
| `aux_dopamine`      | `dopamine`       | folded into `dopamine` (secondary reward channel) |
| `cortisol`          | `norepinephrine` | direct; norepinephrine models stress/arousal |
| `acetylcholine`     | `acetylcholine`  | direct |
| `tempo`             | *(none)*         | intentionally not mapped: it is a step/timebase policy, and neuromod's step has no timebase input |
| *(none)*            | `serotonin`      | left at `0.0`; no hybrid-fusion source |

`tempo` is dropped from the conversion rather than forced onto an unrelated
channel. `serotonin` has no hybrid-fusion source and stays at the neuromod
default of `0.0`. This conversion is internal to the adapter; no neuromod type
appears in any public signature.

---

## 4. GgufLoader / GgufLayout

```rust
pub trait GgufLoader {
    fn load(&self, path: &str) -> Result<GgufLayout>;
}

pub struct GgufLayout {
    pub architecture: String,
    pub tensor_count: usize,
}
```

`GgufLoader` is an extension point for loading checkpoint files. It is **not**
called by the forward pipeline — it exists so backend implementations can
provide a uniform entry point for loading weights.

### Implementing GgufLoader

```rust
use hybrid_fusion::{GgufLoader, GgufLayout, HybridError, Result};

struct MyGgufLoader;

impl GgufLoader for MyGgufLoader {
    fn load(&self, path: &str) -> Result<GgufLayout> {
        // Parse your GGUF file at `path`
        // Return Err(HybridError::ModelLoad { path, reason }) on failure
        // Return Err(HybridError::GgufParse(...)) for parse errors
        // Return Err(HybridError::UnsupportedFormat(...)) for unknown formats

        let file = std::fs::File::open(path)
            .map_err(|e| HybridError::ModelLoad {
                path: path.to_string(),
                reason: e.to_string(),
            })?;

        // ... parse headers, validate magic bytes, count tensors ...

        Ok(GgufLayout {
            architecture: "my-arch".to_string(),
            tensor_count: 42,
        })
    }
}
```

### When to use it

- Loading transformer weights into your `Transformer` implementation.
- Loading SNN weight matrices into your `SpikingNetwork` implementation.
- Validating that a checkpoint file matches the expected architecture
  before constructing your backend types.

Typical wiring:

```rust
let loader = MyGgufLoader;
let layout = loader.load("model.gguf")?;
assert_eq!(layout.architecture, "my-arch");
// Use layout info to construct your Transformer and SpikingNetwork
```

---

## 5. SafetensorsLoader / SafetensorsLayout

```rust
pub trait SafetensorsLoader {
    fn load(&self, path: &str) -> Result<SafetensorsLayout>;
}

pub struct SafetensorsLayout {
    pub architecture: String,
    pub tensor_count: usize,
    pub tensors: Vec<TensorManifestEntry>,
}

pub struct TensorManifestEntry {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub shard: Option<String>,
    pub role: TensorRole,
    pub labels: Vec<String>,
}
```

`SafetensorsLoader` is the Safetensors counterpart of [`GgufLoader`](#4-ggufloader--gguflayout).
It is **not** called by the forward pipeline — it exists so backend
implementations can inventory a `.safetensors` file or a Hugging Face shard
index (`model.safetensors.index.json`) without this crate parsing bytes.

Concrete header parse, mmap, and payload extract stay in
[`engram-parser`](https://github.com/rmems/engram-parser) (off-by-default
`safetensors` feature). Do **not** add a Safetensors crate dependency here.

### Implementing SafetensorsLoader

```rust
use hybrid_fusion::{
    Dtype, HybridError, Result, SafetensorsLayout, SafetensorsLoader,
    TensorManifestEntry, TensorRole,
};

struct MySafetensorsLoader;

impl SafetensorsLoader for MySafetensorsLoader {
    fn load(&self, path: &str) -> Result<SafetensorsLayout> {
        // Parse headers at `path` in engram-parser — not in this crate.
        // Return Err(HybridError::ModelLoad { path, reason }) on I/O failure
        // Return Err(HybridError::SafetensorsParse(...)) for header errors
        // Return Err(HybridError::UnsupportedFormat(...)) for unknown formats
        // Return Err(HybridError::MissingTensor { name, path }) if a required
        //   tensor is absent from the inventory

        let file = std::fs::File::open(path)
            .map_err(|e| HybridError::ModelLoad {
                path: path.to_string(),
                reason: e.to_string(),
            })?;

        // ... inspect header JSON, shard index, count tensors ...
        let _ = file;

        let tensors = vec![TensorManifestEntry::new(
            "model.layers.0.mlp.gate.weight".into(),
            Dtype::F16,
            vec![8, 16],
            Some("model-00001-of-00002.safetensors".into()),
            vec![],
        )];
        Ok(SafetensorsLayout::new("my-arch".into(), tensors))
    }
}
```

### When to use it

- Header-only inspect of a Safetensors checkpoint or HF shard index.
- Building a deterministic tensor manifest (name, dtype, shape, shard refs).
- Discovering MoE router / expert **candidates** via [`TensorRole`] before
  constructing an `ExpertRouter` (issues #22 / #24 / #26).
- Validating that a checkpoint matches the expected architecture before
  constructing backend types.

Typical wiring:

```rust
let loader = MySafetensorsLoader;
let layout = loader.load("model.safetensors")?;
assert_eq!(layout.architecture, "my-arch");
assert_eq!(layout.tensor_count, layout.tensors.len());
let routers = layout.entries_with_role(TensorRole::Router);
// Use router/expert candidates to wire ExpertRouter / dry-run planner
```

**Manifest metadata vs runtime `Tensor`:** checkpoint inventory uses
`TensorManifestEntry::shape` as reported by the Safetensors header. Zero-length
dimensions and rank-0 shapes (`[]`) are valid in that metadata; they are **not**
subject to the runtime rule that [`Tensor::from_vec`](../src/tensor.rs) rejects
zero-extent axes. After JSON deserialization, `SafetensorsLayout.tensor_count` is
always recomputed from `tensors.len()` so a stale count in the wire format cannot
desync the manifest.

### TensorRole (MoE candidate discovery)

`TensorRole` is a **name-heuristic** classifier — no parsing, no family
adapters. `TensorManifestEntry::new` fills `role` from the tensor name:

| Role | Typical names |
|------|----------------|
| `Router` | `…mlp.gate.weight`, `…block_sparse_moe.gate…`, `…moe_router…`, any component containing `router` |
| `ExpertWeight` | `…experts.{i}…` (wins over `gate` / `router` so expert `gate_proj` stays expert) |
| `Attention` | `self_attn`, `SelfAttention`, `q_proj`, … |
| `Embedding` | `embed_tokens`, `lm_head`, T5-style `shared`, … |
| `Norm` | `layernorm`, `rms_norm`, `*.norm.*`, GPT-style `ln_1` / `ln_f` / `ln_*` |
| `Other` (default) | dense FFN `gate_proj` / `up_proj`, unknown names |

Downstream:

- **#22 `ExpertRouter`** — locate gate tensors to bind a checkpoint-backed router.
- **#26 pure MoE math** — file-free; roles are unused until a real gate matmul
  backend lands in `cortex-tensor` + `engram-parser`.
- **#24 `HybridStagePlanner`** — classify stage names (router vs expert vs
  attention) for dry-run precision tiers without loading weights.

`Dtype` is the Safetensors header vocabulary (`F32`, `BF16`, `BOOL`, `U16`,
`F8_E4M3`, `C64`, …), not a precision-planning policy.

---

## 6. How construction and the forward pipeline work

Prefer `HybridNetwork::try_new`. It compares `HybridConfig` against
backend-reported sizes **without** calling `Transformer::hidden_states` or
`SpikingNetwork::step`:

| Config field | Backend method | Error on mismatch or zero |
|--------------|----------------|---------------------------|
| `transformer.dim` | `Transformer::dim()` | `HybridError::ConfigMismatch` (`field = "transformer.dim"`) |
| `transformer.max_seq_len` | `Transformer::max_seq_len()` | `HybridError::ConfigMismatch` (`field = "transformer.max_seq_len"`) |
| `snn_input_channels` | `SpikingNetwork::num_channels()` | `HybridError::ConfigMismatch` (`field = "snn_input_channels"`) |

`HybridNetwork::new` still exists as an unvalidated pre-1.0 compatibility
constructor. It does not panic, does not return `Result`, and does not prove
the backends match the config. New code should use `try_new`.

When you call `HybridNetwork::forward`, this is what happens internally
(see [`src/hybrid.rs`](../src/hybrid.rs)):

```
token_ids: &[u32]
     |
     |  1. Validate: non-empty, len <= max_seq_len()
     |     Validate: snn.num_channels() > 0  (else ZeroSnnChannels)
     v
     |  2. Transformer::hidden_states(token_ids)
     v
Tensor [seq_len, dim]  or  Tensor [dim]
     |
     |  3. Preflight: rank ∈ {1,2}, seq/dim/data length, all-finite
     v
     |  4. pool_embedding: mean-pool across seq dimension -> Vec<f32> of len dim
     |     (reject non-finite pooled values)
     |  5. embed_to_stimuli_with_width:
     |       mean_pool -> resize_to(snn_width) -> tanh squash
     |     (reject non-finite stimuli)
     v
stimuli: Vec<f32>     len == snn.num_channels(),  all values in [-1, 1]
     |
     |  6. SpikingNetwork::step(&stimuli, &modulators)
     |     (only after every preflight check passed; global_step still 0 on error)
     v
fired_neurons: Vec<usize>
```

Step 4 is handled by `pool_embedding` in `src/hybrid.rs` to produce the final
`HybridOutput::embedding`. It mean-pools the hidden-state tensor across the
sequence dimension (if 2-D) to a vector of length `transformer.dim()`.

Step 5 is handled by `projector::embed_to_stimuli_with_width`
(see [`src/projector.rs`](../src/projector.rs)) to produce the SNN stimuli via
these **independent** internal steps (it performs its own mean-pool; it does
**not** reuse the Step 4 embedding vector):

1. **Mean-pool**: If the tensor is 2-D `[seq, dim]`, average across the
   sequence dimension to get a `[dim]` vector. If 1-D, use as-is.
2. **Resize**: Adapt the pooled vector to `snn.num_channels()` length.
   If the pooled vector is longer, it is downsampled via block averaging.
   If shorter, it is zero-padded.
3. **Tanh squash**: Every value is passed through `tanh`, guaranteeing
   the output is in `[-1.0, 1.0]`.

In other words, mean-pooling runs twice on the same hidden tensor — once for
the `embedding` output and once inside the projector for `stimuli`. Backend
implementers should not assume a single shared pooled vector feeds both.

The pipeline returns a `HybridOutput`:

```rust
pub struct HybridOutput {
    pub embedding: Vec<f32>,    // pooled hidden state, len == transformer.dim()
    pub stimuli: Vec<f32>,      // tanh-squashed, len == snn.num_channels()
    pub fired_neurons: Vec<usize>,
    pub num_neurons: usize,     // output population; fired IDs are < this
    pub global_step: u64,
    // MoE reverse-path fields (None on ANN→SNN-only forward):
    pub expert_weights: Option<Vec<f32>>,
    pub selected_experts: Option<Vec<usize>>,
    pub routing_entropy: Option<f32>,
}
```

---

## 7. Tensor shape conventions

`Tensor` is a lightweight owned type: `data: Vec<f32>` + `shape: Vec<usize>`
(see [`src/tensor.rs`](../src/tensor.rs)).

### Rules

- **All dimensions must be `> 0`.** `Tensor::from_vec` and `Tensor::zeros`
  panic on zero-dim shapes. This is a hard invariant.
- **`data.len()` must equal the product of shape dimensions.** Mismatches
  panic at construction time.
- **1-D tensors** `[d]` represent flat vectors. The projector uses them
  directly.
- **2-D tensors** `[rows, cols]` represent matrix layouts. The projector
  interprets them as `[seq_len, hidden_dim]` and mean-pools across rows.

### Creating tensors

```rust
use hybrid_fusion::Tensor;

// 1-D: a single embedding of dimension 128
let t = Tensor::from_vec(vec![0.0; 128], &[128]);

// 2-D: 4 tokens, each with 128-dim hidden state
let t = Tensor::from_vec(vec![0.0; 4 * 128], &[4, 128]);

// Zeros (same rules apply)
let t = Tensor::zeros(&[8, 256]);
```

---

## 8. Minimal working example

This example implements both traits with trivial math and wires them into
`HybridNetwork`. It compiles against the `hybrid-fusion` public API.

```rust
use hybrid_fusion::{
    HybridConfig, HybridError, HybridNetwork, HybridOutput, NeuroModulators,
    Result, SpikingNetwork, Tensor, Transformer,
};

// ---------------------------------------------------------------------------
// Transformer: returns per-token hidden states as a 2-D tensor
// ---------------------------------------------------------------------------

struct MyTransformer {
    dim: usize,
    max_seq_len: usize,
}

impl Transformer for MyTransformer {
    fn hidden_states(&self, token_ids: &[u32]) -> Tensor {
        let seq_len = token_ids.len();
        // Dummy: multiply each token id by a small float to fill the tensor.
        // A real backend would run attention, FFN layers, etc.
        let mut data = Vec::with_capacity(seq_len * self.dim);
        for &tok in token_ids {
            let base = tok as f32 * 0.01;
            for j in 0..self.dim {
                data.push(base + j as f32 * 0.001);
            }
        }
        Tensor::from_vec(data, &[seq_len, self.dim])
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn param_count(&self) -> usize {
        // Placeholder — a real backend would report actual weight count
        self.dim * self.dim * 12
    }
}

// ---------------------------------------------------------------------------
// SpikingNetwork: simple threshold-based neuron model
// ---------------------------------------------------------------------------

struct MySnn {
    num_channels: usize,
    /// Membrane potentials, one per channel
    potentials: Vec<f32>,
    /// Firing threshold
    threshold: f32,
}

impl MySnn {
    fn new(num_channels: usize, threshold: f32) -> Self {
        Self {
            num_channels,
            potentials: vec![0.0; num_channels],
            threshold,
        }
    }
}

impl SpikingNetwork for MySnn {
    fn step(&mut self, stimuli: &[f32], modulators: &NeuroModulators) -> Result<Vec<usize>> {
        if stimuli.len() != self.num_channels {
            return Err(HybridError::SnnStep(format!(
                "stimuli length {} != num_channels {}",
                stimuli.len(),
                self.num_channels
            )));
        }
        let mut fired = Vec::new();

        for (i, &input) in stimuli.iter().enumerate() {
            // Leaky integrate: decay old potential, add new input
            self.potentials[i] = self.potentials[i] * 0.9 + input;

            // Scale threshold by cortisol (stress raises the bar)
            let effective_threshold = self.threshold * (1.0 + modulators.cortisol * 0.5);

            // Dopamine lowers the threshold (reward increases excitability)
            let effective_threshold =
                effective_threshold * (1.0 - modulators.dopamine * 0.3);

            if self.potentials[i] > effective_threshold {
                fired.push(i);
                self.potentials[i] = 0.0; // reset after firing
            }
        }

        Ok(fired)
    }

    fn num_channels(&self) -> usize {
        self.num_channels
    }
}

// ---------------------------------------------------------------------------
// Wire it together
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let config = HybridConfig::tiny();

    let transformer = MyTransformer {
        dim: config.transformer.dim,             // 128
        max_seq_len: config.transformer.max_seq_len, // 64
    };

    let snn = MySnn::new(config.snn_input_channels, 0.5); // 64 channels

    let mut net = HybridNetwork::try_new(transformer, snn, config)?;

    // Forward pass with default modulators
    let output: HybridOutput = net.forward(&[1, 2, 3, 4], None)?;
    println!("embedding len: {}", output.embedding.len()); // 128
    println!("stimuli  len: {}", output.stimuli.len());    // 64
    println!("fired neurons: {:?}", output.fired_neurons);
    println!("global step:   {}", output.global_step);     // 1

    // Forward pass with custom modulators
    let mods = NeuroModulators {
        dopamine: 0.8,
        cortisol: 0.2,
        acetylcholine: 0.6,
        tempo: 1.2,
        aux_dopamine: 0.0,
    };
    let output2 = net.forward(&[10, 20, 30], Some(mods))?;
    println!("fired: {:?}", output2.fired_neurons);
    println!("step:  {}", output2.global_step); // 2

    Ok(())
}
```

### Key points in this example

- `MyTransformer::hidden_states` returns a **2-D** tensor `[seq_len, dim]`,
  matching `Transformer::dim()`. The projector mean-pools and resizes
  automatically.
- `MySnn::new` takes `num_channels` which equals `config.snn_input_channels`
  — this is independent from the transformer dim.
- `MySnn::step` uses `NeuroModulators` to modulate the firing threshold:
  dopamine lowers it (more excitable), cortisol raises it (more conservative).
- After firing, the membrane potential resets to `0.0` (leaky
  integrate-and-fire style).
- `HybridNetwork::try_new` validates the transformer, SNN, and config. The
  config's `snn_input_channels` must match `snn.num_channels()`, and
  transformer `dim` / `max_seq_len` must match the trait reports. Zero
  capacities are rejected. `HybridNetwork::new` skips this check (compatibility).

---

## 9. Common pitfalls

### Config vs backend mismatch at construction

`HybridNetwork::try_new` returns `HybridError::ConfigMismatch` when a
configured size is zero or disagrees with the backend:

```
configuration mismatch for transformer.dim: configured 64, backend 128
```

**Fix:** Set `HybridConfig` from the same values your `Transformer` /
`SpikingNetwork` report (`dim()`, `max_seq_len()`, `num_channels()`).

### Shape mismatch: `hidden_states` dim vs `dim()`

If your 2-D tensor's second dimension doesn't match `dim()`, the pipeline
returns `HybridError::HiddenStateDim { expected, got }` (not a Sentry
runtime failure):

```
hidden-state dimension mismatch: expected 128, got 64
```

Rank-2 `shape[0]` must also equal `token_ids.len()` (`HiddenStateSeqLen`).
Rank 0 / 3+ is `HiddenStateRank`. NaN / ±Inf is `NonFinite`.

**Fix:** Ensure `Tensor::from_vec(data, &[seq_len, self.dim])` uses the same
`dim` value as `fn dim(&self) -> usize`, and `seq_len == token_ids.len()`.

### Zero-dim tensors panic

`Tensor::from_vec` and `Tensor::zeros` **panic** (not error) if any shape
dimension is `0`:

```
Tensor::from_vec: shape dimensions must be > 0, got [0]
```

**Fix:** Never return a tensor with a zero-length dimension. If your
transformer produces no output for an edge case, return a 1-D tensor of
the correct dim with zeroed data instead.

### Empty token_ids

`HybridNetwork::forward` returns `HybridError::InputLengthMismatch` if
`token_ids` is empty. This check happens **before** your `hidden_states`
is called.

### Input exceeds max_seq_len

If `token_ids.len() > max_seq_len()`, the pipeline returns
`HybridError::InputLengthMismatch` before calling `hidden_states`.

### SNN input length mismatch

The projector always produces `snn.num_channels()` values. If your `step`
implementation expects a different length, you'll get a length mismatch
or silent data corruption. The example above returns
`Err(HybridError::SnnStep(...))` so callers can recover instead of panicking.

### Stimuli are always in [-1, 1]

The projector applies `tanh` to all stimuli values. Your SNN should expect
inputs in this range. Don't assume raw embedding magnitudes — they are
squashed before reaching you.

### Modulator defaults

If the caller passes `None` for modulators, the pipeline uses
`NeuroModulators::default()`:

```rust
dopamine: 0.5, cortisol: 0.0, acetylcholine: 0.5, tempo: 1.0, aux_dopamine: 0.0
```

Design your neuron dynamics to work sensibly with these neutral values.

### SpikingNetwork::step returns Err

If your SNN step fails, propagate the error with
`Err(HybridError::SnnStep("reason".into()))`. The pipeline will forward
it to the caller. Don't panic — `HybridNetwork::forward` expects a
`Result`.

---

## 10. Error reference

All errors come from [`src/error.rs`](../src/error.rs):

| Variant | When |
|---------|------|
| `InputLengthMismatch { expected, got }` | Empty input or exceeds `max_seq_len` |
| `ConfigMismatch { field, configured, backend }` | `try_new`: config vs backend dim / max sequence / SNN channels (including zeros) |
| `HiddenStateRank { got }` | Hidden-state rank is not 1 or 2 |
| `HiddenStateSeqLen { expected, got }` | Rank-2 `shape[0]` ≠ `token_ids.len()` |
| `HiddenStateDim { expected, got }` | Hidden width ≠ `Transformer::dim()` |
| `HiddenStateDataLen { expected, got }` | Backing `data.len()` ≠ layout product |
| `NonFinite { stage, index }` | NaN or ±Inf in hidden / embedding / stimuli |
| `ZeroSnnChannels` | `SpikingNetwork::num_channels()` is 0 |
| `InvalidConfig(String)` | Other host/config contract failures |
| `SnnStep(String)` | SNN internal error during `step` (Sentry when enabled) |
| `ModelLoad { path, reason }` | File I/O failure during loading |
| `MissingTensor { name, path }` | Expected tensor not found in checkpoint |
| `GgufParse(String)` | GGUF file parse failure |
| `SafetensorsParse(String)` | Safetensors header / shard-index parse failure |
| `UnsupportedFormat(String)` | Unknown checkpoint format |
| `Io(...)` | std::io error (propagated via `From`) |
| `Json(...)` | serde_json error (propagated via `From`) |

---

## 11. Reverse path: SpikeActivity + ExpertRouter + ReverseHybridPath

The **ANN → SNN** path above is complete for `HybridNetwork::forward`. A
second path is being extracted from research (corinth-canal):

```text
SpikeActivity → (ProjectionMode / projector — issue #25)
             → embedding → ExpertRouter::route → ExpertRouteOutput
```

### `SpikeActivity`

Pure data bag (not a trait). Fields mirror corinth-canal funnel / projector
inputs without GIF dynamics:

```rust
pub struct SpikeActivity {
    pub spike_train: Vec<Vec<usize>>,
    pub potentials: Vec<f32>,
    pub iz_potentials: Vec<f32>,
}
```

Helper: `SpikeActivity::from_fired(fired, n_neurons) -> Result` for one-step
tests. Rejects `n_neurons == 0` and any fired index `>= n_neurons`; empty
`fired` with `n_neurons > 0` is valid. Use `snn.num_neurons()` or
`HybridOutput.num_neurons`, never input width, for this dimension and the
reverse host's `n_neurons`.

### `ExpertRouter`

```rust
pub trait ExpertRouter {
    fn num_experts(&self) -> usize;
    fn top_k(&self) -> usize;
    fn route(&mut self, embedding: &[f32]) -> Result<ExpertRouteOutput>;
}
```

`ExpertRouteOutput` carries `expert_weights`, `selected_experts`, and optional
`routing_entropy`. Embedding length is **not** fixed to 2048.

**Weight normalization is enforced by the host.** `ReverseHybridPath::forward_activity`
re-accumulates `expert_weights` in `f64` and requires the sum to be within
`WEIGHT_SUM_TOLERANCE` (`8 * f32::EPSILON`, ~`9.54e-7`) of `1.0`. Inside that band
the weights are renormalized in `f64`; outside it — or for an all-zero
distribution — the call fails with `InvalidConfig` instead of silently rescaling.
The bound is derived from the `f32` unit round-off and does **not** grow with the
expert count, so **accumulate your softmax denominator in `f64`**: an `f32`
denominator drifts by `O(num_experts * 2^-24)` (up to ~`4.8e-2` at
`MAX_REASONABLE_EXPERTS`) and will be rejected at large expert counts. Return a
normalized distribution (e.g. via `softmax`), not raw gate scores.

**Reference stub** (feature `backends`): `StubExpertRouter` returns a uniform
gate distribution and selects `0..top_k`.

**Out of scope for this crate:** Safetensors **parsing / mmap / payload extract**
(and GGUF parse/mmap) stay in `engram-parser`. Family adapters and real gate
matmul stay in sibling crates. The **trait/layout contract**
(`SafetensorsLoader`, `SafetensorsLayout`, `TensorRole`) now lives here,
symmetric to `GgufLoader`.

`HybridNetwork::forward` (ANN → SNN) always leaves MoE fields as `None`.
For the reverse path, use [`ReverseHybridPath`](../src/reverse.rs):

```rust
use hybrid_fusion::{
    ProjectionMode, ReverseHybridPath, SpikeActivity,
    // with --features backends:
    // SyntheticExpertRouter,
};

// let router = my_expert_router; // or SyntheticExpertRouter::new(4, 2)?
// let mut path = ReverseHybridPath::new(
//     ProjectionMode::RateSum,
//     /* n_neurons */ 8,
//     /* embed_dim */ 16,
//     router,
// )?;
// let activity = SpikeActivity::from_fired(&[0, 2], 8)?;
// let out = path.forward_activity(&activity)?;
// assert!(out.expert_weights.is_some());
// assert!(out.selected_experts.is_some());
// assert!(out.stimuli.is_empty());
```

### Plugging in a real MoE backend later

1. Implement `ExpertRouter` with checkpoint-backed gate matmul (engram-parser +
   cortex-tensor — not in this crate).
2. Construct `ReverseHybridPath::new(mode, n_neurons, embed_dim, my_router)`.
3. Call `forward_activity` with live `SpikeActivity` from neuromod / funnel.
4. Do **not** use this path for SAAQ / latent calibration.

**v1 error policy:** reverse-path router/project errors propagate without Sentry
capture (validation-heavy; avoids flooding on bad activity).

### `ProjectionMode` + pure project

```rust
use hybrid_fusion::{
    project_spike_activity, ProjectionMode, SpikeActivity, ExpertRouter,
};

let activity = SpikeActivity::from_fired(&[0, 2], 8)?;
// Dense embedding for MoE ExpertRouter — not SAAQ / latent calibration.
let embedding = project_spike_activity(
    ProjectionMode::RateSum,
    &activity,
    /* n_neurons */ 8,
    /* embed_dim */ 16,
)?;
// router.route(&embedding)?;
```

| Mode | Pure feature vector |
|------|---------------------|
| `RateSum` | per-neuron firing rates only (`n_neurons`) |
| `TemporalHistogram` | time-binned rates only (`n_neurons × 4`) |
| `MembraneSnapshot` | clamped membrane potentials only (`n_neurons`) |
| `SpikingTernary` | identical to RateSum on the pure path; GIF → `neuromod` |

No learned W/b matrix. Embedding is mode features → resize + `tanh` only.

### Pure MoE math (`routing`)

Always available (no feature flag) for [`ExpertRouter`](../src/traits.rs) backends:

| Helper | Role |
|--------|------|
| `synthetic_gate_scores` | partition embedding → per-expert scores (full coverage) |
| `softmax` | normalize scores (sum ≈ 1); `NaN` → `Err`, `+Inf` → mass split across `+Inf` indices only, all `-Inf` → uniform |
| `top_k_indices` | select experts (NaNs sort last) |
| `routing_entropy` | Shannon entropy normalized to `[0, 1]` (`f64` accumulate) |
| `route_synthetic` | all of the above in one call (caps `num_experts`) |

**Feature `backends`:** `SyntheticExpertRouter` implements `ExpertRouter` via
`route_synthetic`. Uniform stub remains `StubExpertRouter`. Without the feature,
use the free functions above with your own type.
