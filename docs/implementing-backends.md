# Implementing a Backend

`hybrid-fusion` is backend-agnostic: it defines the `Transformer`, `SpikingNetwork`,
`GgufLoader`, `ExpertRouter`, `SpikeActivity`, and `NeuroModulators` contracts but
ships no concrete math. This guide explains every trait method, how data flows
through `HybridNetwork::forward`, and provides a minimal compilable example you can
adapt for your own backend.

For authoritative signatures, always refer to [`src/traits.rs`](../src/traits.rs).

---

## Table of Contents

1. [Transformer trait](#1-transformer-trait)
2. [SpikingNetwork trait](#2-spikingnetwork-trait)
3. [NeuroModulators](#3-neuromodulators)
4. [GgufLoader / GgufLayout](#4-ggufloader--gguflayout)
5. [How the forward pipeline works](#5-how-the-forward-pipeline-works)
6. [Tensor shape conventions](#6-tensor-shape-conventions)
7. [Minimal working example](#7-minimal-working-example)
8. [Common pitfalls](#8-common-pitfalls)
9. [Error reference](#9-error-reference)
10. [Reverse path: SpikeActivity + ExpertRouter](#10-reverse-path-spikeactivity--expertrouter)

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
- The returned `Tensor` **must** have one of these shapes:
  - **1-D** `[dim]` — a single pooled embedding vector. This is the simplest
    form; the projector will use it directly.
  - **2-D** `[seq_len, dim]` — per-token hidden states. The projector will
    mean-pool across the sequence dimension, then resize to match the SNN
    input width.
- If the tensor is 2-D, `shape[1]` **must equal** `dim()`. The forward
  pipeline checks this and returns `HybridError::InvalidConfig` on mismatch.
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
tensor shapes returned by `hidden_states`. For 2-D tensors, the pipeline
validates `shape[1] == dim()`.

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
  must be valid for your neuron population. An empty `Vec` means no neurons
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

**Note:** `num_channels` is independent from `Transformer::dim()`. The
transformer might produce a 128-dim embedding while the SNN only has 64
input channels — the projector handles the resize.

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

## 5. How the forward pipeline works

When you call `HybridNetwork::forward`, this is what happens internally
(see [`src/hybrid.rs`](../src/hybrid.rs)):

```
token_ids: &[u32]
     |
     |  1. Validate: non-empty, len <= max_seq_len()
     v
     |  2. Transformer::hidden_states(token_ids)
     v
Tensor [seq_len, dim]  or  Tensor [dim]
     |
     |  3. Validate: if 2-D, shape[1] == dim()
     v
     |  4. pool_embedding: mean-pool across seq dimension -> Vec<f32> of len dim
     |  5. embed_to_stimuli_with_width:
     |       mean_pool -> resize_to(snn_width) -> tanh squash
     v
stimuli: Vec<f32>     len == snn.num_channels(),  all values in [-1, 1]
     |
     |  6. SpikingNetwork::step(&stimuli, &modulators)
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
    pub global_step: u64,
    // MoE reverse-path fields (None on ANN→SNN-only forward):
    pub expert_weights: Option<Vec<f32>>,
    pub selected_experts: Option<Vec<usize>>,
    pub routing_entropy: Option<f32>,
}
```

---

## 6. Tensor shape conventions

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

## 7. Minimal working example

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

    let mut net = HybridNetwork::new(transformer, snn, config);

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
- `HybridNetwork::new` accepts the transformer, SNN, and config. The
  config's `snn_input_channels` must match `snn.num_channels()`.

---

## 8. Common pitfalls

### Shape mismatch: `hidden_states` dim vs `dim()`

If your 2-D tensor's second dimension doesn't match `dim()`, the pipeline
returns `HybridError::InvalidConfig`:

```
invalid configuration: hidden state dim=64 does not match transformer.dim()=128
```

**Fix:** Ensure `Tensor::from_vec(data, &[seq_len, self.dim])` uses the same
`dim` value as `fn dim(&self) -> usize`.

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

## 9. Error reference

All errors come from [`src/error.rs`](../src/error.rs):

| Variant | When |
|---------|------|
| `InputLengthMismatch { expected, got }` | Empty input or exceeds `max_seq_len` |
| `InvalidConfig(String)` | Hidden-state dim doesn't match `dim()` |
| `SnnStep(String)` | SNN internal error during `step` |
| `ModelLoad { path, reason }` | File I/O failure during loading |
| `MissingTensor { name, path }` | Expected tensor not found in checkpoint |
| `GgufParse(String)` | GGUF file parse failure |
| `UnsupportedFormat(String)` | Unknown checkpoint format |
| `Io(...)` | std::io error (propagated via `From`) |
| `Json(...)` | serde_json error (propagated via `From`) |

---

## 10. Reverse path: SpikeActivity + ExpertRouter

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
`fired` with `n_neurons > 0` is valid.

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

**Reference stub** (feature `backends`): `StubExpertRouter` returns a uniform
gate distribution and selects `0..top_k`.

**Out of scope for this crate's trait surface:** GGUF/Safetensors weight load,
family adapters, real gate matmul (see extraction map / sibling crates).

`HybridNetwork::forward` leaves `HybridOutput` MoE fields as `None` until a
reverse-path orchestrator lands (follow-ups #26 / #23).

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
| `softmax` | normalize scores (sum ≈ 1) |
| `top_k_indices` | select experts (NaNs sort last) |
| `routing_entropy` | Shannon entropy normalized to `[0, 1]` (`f64` accumulate) |
| `route_synthetic` | all of the above in one call (caps `num_experts`) |

**Feature `backends`:** `SyntheticExpertRouter` implements `ExpertRouter` via
`route_synthetic`. Uniform stub remains `StubExpertRouter`. Without the feature,
use the free functions above with your own type.
