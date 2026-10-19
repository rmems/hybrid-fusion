# ADR 0001: one ANN↔SNN step contract, owned by hybrid-fusion

- **Status:** Proposed (2026-10-19)
- **Tracking:** rmems/hybrid-fusion#61 (Linear RM-2205); this ADR is Linear RM-2210.
  Twin removal issue: rmems/cortex-tensor#137. Related: hybrid-fusion#62 (ResultTrack),
  hybrid-fusion#44 (neuromod dependency exception), hybrid-fusion#20 (epic).
- **Trees cited:** hybrid-fusion `506ff5b`, cortex-tensor `a0fd18e`
  (open cortex-tensor PRs #130, #131 read from their GitHub diffs).
- **Scope:** contracts and placement only. No code changes ship with this ADR.
  Spikenaut-SNN internals and benchmarking (combine-for-AI) are out of scope.

## 1. Context

Two crates each define a contract for stepping a spiking network from ANN-side
signals, and each ships a `neuromod` adapter behind an optional feature.

### 1.1 cortex-tensor `src/snn/` (the ANN reference backend)

| Item | Location | Shape |
|---|---|---|
| `SnnCapabilities` | `src/snn/mod.rs:45-56` | `{backend_name, frozen_evaluation, plasticity, caller_rng, neuromodulation}` — a second capability vocabulary |
| `SnnStepOutput` | `src/snn/mod.rs:60-64` | `{spikes: Vec<usize>}` in `0..channels` |
| `SnnBackend` | `src/snn/mod.rs:71-91` | `channels()`, `step(&[f32]) -> Result<SnnStepOutput>`, `step_frozen(..)`, `reset()`, `capabilities()`; failure-atomic on rejected input |
| `SnnEncoder` | `src/snn/mod.rs:99-109` | `encode(&mut self, ann_signal, channels) -> Result<Vec<f32>>`, `reset()` |
| `SnnDecoder` | `src/snn/mod.rs:112-115` | `decode(&SnnStepOutput, channels) -> Vec<f32>` |
| `RateEncoder` | `src/snn/mod.rs:121-143` | gain × tile |
| `SpikeCountDecoder` | `src/snn/mod.rs:148-160` | per-channel spike count |
| `SpikingMoeRouter<B, E, D>` | `src/snn/mod.rs:176-317` | gate scores → encode → step → decode → `gate * (1 + spikes)` → top-k; `hidden = embedding * Σ spikes` |
| `NeuromodNetwork` | `src/snn/neuromod_adapter.rs:21-106` | 106-line `SnnBackend` over `neuromod::SpikingNetwork`; `step_frozen` delegates to `step`; `caller_rng: true` hard-coded; `NeuroModulators::default()` every step |
| `AxonEncoder<E>` | `src/snn/axon_adapter.rs:33-76` | `SnnEncoder` over `axon_encoder::Encoder::encode_step` |
| Errors | `src/error.rs:75-87` | `SnnBackend`, `SnnChannelMismatch`, `SnnNan`, `SnnSpikeOutOfRange` |
| Features | `Cargo.toml:37-39, 56-57` | `neuromod = "0.7"`, `axon-encoder = "0.5"`, both optional |

Width rule: `channels()` is both stimulus width and spike-index space
(`mod.rs:72-75`); the adapter therefore requires neuron count == channel count
(`neuromod_adapter.rs:49-61`). One expert ↔ one channel (`mod.rs:194-203`).

### 1.2 hybrid-fusion `src/` (the orchestrator)

| Item | Location | Shape |
|---|---|---|
| `SpikingNetwork` | `src/traits.rs:55-98` | `step(&mut self, stimuli, &NeuroModulators) -> Result<Vec<usize>>`, `num_channels()`, `num_neurons()` (default `= num_channels`), `reset() -> Result<()>` (default: unsupported error), `capabilities() -> BackendCapabilities` |
| `NeuroModulators` | `src/traits.rs:100-119` | `{dopamine, cortisol, acetylcholine, tempo, aux_dopamine}` |
| `ExpertRouter` / `ExpertRouteOutput` | `src/traits.rs:334-369` | `route(&mut self, embedding) -> Result<ExpertRouteOutput{expert_weights (normalized), selected_experts, routing_entropy}>` |
| `SpikeActivity` | `src/traits.rs:261-302` | reverse-path spike/membrane bag |
| `BackendCapabilities` | `src/capabilities.rs:145-335` | `ann()/snn()` builders, `with_channels`, `with_num_neurons`, `with_stateful`, `with_reset`, `with_features([RequiredFeature])` |
| `PortSpec` / `DimSpec` | `src/plan.rs:150-170` | `{dtype: Option<Dtype>, dims: Vec<DimSpec>}` |
| `ExecutionDomain` | `src/plan.rs:129-141` | `Auto, Ann, Snn, Moe, Adapter` |
| `HybridError` | `src/error.rs:32-107` | incl. `InputLengthMismatch`, `NonFinite{stage,index}`, `ZeroSnnChannels`, `SnnStep(String)`, `ConfigMismatch` |
| `HybridNetwork::forward` | `src/hybrid.rs:125-190` | pool → project to `snn_input_channels` → `step` → bounds-check fired IDs (`hybrid.rs:173`) |
| `NeuromodSnn` | `src/neuromod_adapter.rs:171-492` | 1,489-line adapter (incl. tests): explicit `num_channels` × `num_neurons`, seeded weights, owned `StdRng` via `step_with_rng`, `reset` restores initial weights/thresholds, 64 MiB matrix budget, exhaustive `StepError` mapping, explicit modulator vocabulary mapping |
| Feature | `Cargo.toml:51-68` | `neuromod = "0.7"` optional, authorized by #44 |

### 1.3 Why this is the boundary blur

cortex-tensor is the ANN reference backend. By defining `SnnBackend` and composing
`SpikingMoeRouter`, it both specifies the SNN side of the boundary and performs
hybrid routing, so ANN and hybrid code paths live in the same crate and the same
`MoeOutput` type. hybrid-fusion already owns orchestration and the capability
vocabulary (#20, AGENTS.md "Crate boundaries"), and its adapter already
translates cortex's `SnnCapabilities` into `BackendCapabilities`
(`neuromod_adapter.rs:463-491`). Two contracts means the #91–#93 fixes would be
ported twice (issue #61 "Sequencing").

## 2. Decision

**hybrid-fusion owns the single ANN↔SNN step contract.** The contract is the
existing `SpikingNetwork` trait, extended with the two things cortex's
`SnnBackend` had that it lacks (a frozen step and an explicit encode/decode pair),
plus a hybrid spiking router that lives on the orchestrator side. cortex-tensor
keeps no spiking types.

### 2.1 Unified trait set (Rust sketches)

```rust
// src/traits.rs — the step contract (existing, extended)
pub trait SpikingNetwork {
    /// Learning-enabled tick. Must be failure-atomic: a rejected input leaves
    /// backend state unmodified.
    fn step(&mut self, stimuli: &[f32], modulators: &NeuroModulators) -> Result<Vec<usize>>;

    /// Held-out tick: no plasticity-controlled state (weights, eligibility,
    /// thresholds) may change. Default delegates to `step`; a backend that
    /// does so MUST NOT advertise the `frozen_evaluation` feature.
    fn step_frozen(&mut self, stimuli: &[f32], modulators: &NeuroModulators) -> Result<Vec<usize>> {
        self.step(stimuli, modulators)
    }

    fn num_channels(&self) -> usize;
    fn num_neurons(&self) -> usize { self.num_channels() }
    fn reset(&mut self) -> Result<()> { Err(HybridError::SnnStep("reset is not supported".into())) }
    fn capabilities(&self) -> BackendCapabilities { /* unchanged default */ }
}
```

```rust
// src/encoding.rs (new) — ANN→SNN and SNN→ANN boundary adapters,
// ported from cortex-tensor src/snn/mod.rs:99-115 with HybridError.
pub trait SpikeEncoder {
    /// Exactly `channels` stimuli, or Err. Stateful (phase/window): one call is
    /// one tick, and state advance is NOT failure-atomic w.r.t. later stages.
    fn encode(&mut self, ann_signal: &[f32], channels: usize) -> Result<Vec<f32>>;
    fn reset(&mut self) {}
    /// Declared on the plan port (see §5). Default: rate, 1 timestep, unbounded.
    fn encoding(&self) -> SpikeEncoding { SpikeEncoding::default() }
}

pub trait SpikeDecoder {
    /// Exactly `num_neurons` readouts from one tick of fired IDs.
    fn decode(&self, fired: &[usize], num_neurons: usize) -> Result<Vec<f32>>;
    fn decoding(&self) -> SpikeDecoding { SpikeDecoding::default() }
}

pub struct RateEncoder { pub gain: f32 }      // cortex RateEncoder, unchanged semantics
pub struct SpikeCountDecoder;                 // cortex SpikeCountDecoder, now fallible
```

Note the one deliberate change from cortex: `decode` takes `num_neurons`, not
`channels`. hybrid-fusion separates input width from output population
(`traits.rs:61-69`); cortex collapsed them (`mod.rs:72-75`). Backends with
`num_neurons != num_channels` keep working; the one-expert-one-channel case is
`num_neurons == num_channels == num_experts`.

```rust
// src/routing.rs or src/spiking_router.rs (new) — hybrid MoE routing host,
// ported from cortex SpikingMoeRouter (mod.rs:176-317) with the #130 fixes.
pub struct SpikingExpertRouter<R: ExpertRouter, S: SpikingNetwork, E: SpikeEncoder, D: SpikeDecoder> {
    router: R, snn: S, encoder: E, decoder: D, spike_boost: f32,
}
impl<..> SpikingExpertRouter<..> {
    pub fn new(router: R, snn: S, encoder: E, decoder: D) -> Result<Self>; // rejects num_neurons < num_experts
    pub fn with_spike_boost(self, boost: f32) -> Result<Self>;             // finite, >= 0
    pub fn forward(&mut self, embedding: &[f32], m: &NeuroModulators) -> Result<SpikingRouteOutput>;
    pub fn forward_frozen(&mut self, embedding: &[f32], m: &NeuroModulators) -> Result<SpikingRouteOutput>;
    pub fn reset(&mut self) -> Result<()>;  // encoder.reset(); snn.reset()
}
pub struct SpikingRouteOutput {
    pub route: ExpertRouteOutput,   // normalized weights + selection, same invariants as ExpertRouter
    pub fired_neurons: Vec<usize>,
    pub num_neurons: usize,
    pub hidden_gain: f32,           // bounded, see §6 (#130 item 2)
    pub track: ResultTrack,         // always Hybrid, see §5
}
```

`SpikingExpertRouter` is a host like `ReverseHybridPath`, not a `HybridNetwork`
generic (AGENTS.md: do not fold reverse-path hosts into `HybridNetwork`).

### 2.2 Error contract

All boundary errors are `HybridError`. New variants (names follow cortex PR #130
so the ported tests keep their assertions):

| Condition | Variant |
|---|---|
| stimulus/readout length ≠ expected | `InputLengthMismatch { expected, got }` (existing) |
| NaN in encoder stimulus / decoder readout | `NonFinite { stage, index }` (existing; add `ForwardValueStage::EncoderStimulus`, `DecoderReadout`) |
| backend rejected the tick | `SnnStep(String)` (existing) |
| fired ID ≥ `num_neurons` | `SnnSpikeOutOfRange { neuron, num_neurons }` (new; today `hybrid.rs:173` reports this as `SnnStep` text) |
| same neuron fired twice in one tick | `SnnDuplicateSpike { neuron }` (new, #130) |
| negative or ±Inf readout | `SnnInvalidReadout { index, value }` (new, #130) |
| `num_neurons < num_experts` at construction | `ConfigMismatch { field: FIELD_SNN_NUM_NEURONS, configured, backend }` (existing variant, new field constant) |
| invalid `spike_boost` | `InvalidConfig` (existing) |

Ordering rule (kept from cortex `mod.rs:213-239`): everything that can be checked
*before* `step` is checked before `step`, so a rejected forward never consumes a
tick. Errors after `step` have consumed a tick; `reset()` restores a clean epoch.

### 2.3 Mapping: every cortex-tensor `snn` item → new home

| cortex-tensor item | New home | Notes |
|---|---|---|
| `SnnBackend` | `hybrid_fusion::SpikingNetwork` | `channels()` → `num_channels()`/`num_neurons()`; `step_frozen` added; `reset` already fallible |
| `SnnBackend::step(&[f32])` | `SpikingNetwork::step(&[f32], &NeuroModulators)` | modulators become an explicit argument instead of adapter state (`neuromod_adapter.rs:23-24`) |
| `SnnStepOutput { spikes }` | `Vec<usize>` fired IDs | removed as a type; the index space is `0..num_neurons` |
| `SnnCapabilities` | `BackendCapabilities::snn(..)` + `RequiredFeature`s `step`, `frozen_evaluation`, `plasticity`, `caller_rng`, `neuromodulation` | removed as a type; mapping already exists at `hybrid-fusion/src/neuromod_adapter.rs:476-491` |
| `SnnEncoder` | `hybrid_fusion::SpikeEncoder` | plus `encoding()` port metadata |
| `SnnDecoder` | `hybrid_fusion::SpikeDecoder` | fallible; sized by `num_neurons` |
| `RateEncoder` | `hybrid_fusion::RateEncoder` | unchanged semantics |
| `SpikeCountDecoder` | `hybrid_fusion::SpikeCountDecoder` | unchanged semantics, fallible |
| `SpikingMoeRouter<B,E,D>` | `hybrid_fusion::SpikingExpertRouter<R,S,E,D>` | composes an `ExpertRouter` instead of a concrete `MoeRouter`; carries #130 |
| `NeuromodNetwork` | **removed**; `hybrid_fusion::NeuromodSnn` is the one adapter (§4) | #131 behaviours ported into `NeuromodSnn` (§6) |
| `AxonEncoder<E>` | **removed from cortex**; not re-homed in hybrid-fusion by this ADR | would need an `axon-encoder` dependency exception like #44. Open question §8 |
| `CortexError::{SnnBackend, SnnChannelMismatch, SnnNan, SnnSpikeOutOfRange}` | removed; `HybridError` per §2.2 | |
| features `neuromod`, `axon-encoder` and deps (`Cargo.toml:37-39,56-57`) | removed from cortex-tensor | hybrid-fusion keeps `neuromod` |
| `src/snn/tests.rs` (13 tests on `a0fd18e`, +7 in #130, +~10 in #131) | ported to hybrid-fusion alongside the relocated code | assertions re-targeted to `HybridError` |
| README rows citing `snn::` (`README.md:100,128,145,183,241`) | rewritten to point at hybrid-fusion | |

## 3. What stays in cortex-tensor

Only the ANN-side ingress/egress at the hidden-state level:

- `adapter::HiddenStateBuffer { data: Vec<f32>, shape, dtype }` (`src/adapter.rs:86-94`)
  as the owned interchange buffer, plus `export_hidden` / `import_hidden`
  (`src/stage.rs:1108-1152`) and `AdapterCapabilities::hidden_state_io`
  (`src/adapter.rs:107`).
- `MoeRouter` and its dense routing (`gate_scores`, `route_top_k`) remain pure ANN.

cortex-tensor exports hidden states; hybrid-fusion pools/projects/encodes them,
steps the SNN, decodes, and (if routing) re-enters the ANN through
`ExpertRouter`. cortex-tensor never names a spike, a channel, or a neuromodulator.
`CortexError` keeps no `Snn*` variant.

## 4. The single neuromod adapter lives in hybrid-fusion

Decision: keep `hybrid_fusion::NeuromodSnn` (`src/neuromod_adapter.rs`) under the
existing #44 exception; delete `cortex_tensor::snn::NeuromodNetwork`.

Why hybrid-fusion and not the `neuromod` crate:

1. **Dependency direction.** hybrid-fusion already optionally depends on
   `neuromod` 0.7. An adapter inside `neuromod` would have to depend on
   hybrid-fusion for `SpikingNetwork`, `NeuroModulators`, `HybridError` and
   `BackendCapabilities`: a cycle unless the traits move to a third crate, which
   #61 does not ask for.
2. **The vocabulary mapping belongs with the vocabulary.** hybrid-fusion defines
   `NeuroModulators {dopamine, cortisol, acetylcholine, tempo, aux_dopamine}`;
   neuromod has `{dopamine, serotonin, acetylcholine, norepinephrine}`. The
   explicit per-field table (`neuromod_adapter.rs:33-61`) is a hybrid-fusion
   contract decision, not a neuron-dynamics decision.
3. **The hybrid-fusion adapter is the more complete one.** Explicit
   `num_channels × num_neurons`, seeded strictly positive weights, owned seeded
   RNG through `step_with_rng`, weight/threshold restore on `reset`, 64 MiB
   matrix budget, exhaustive `StepError` → `HybridError` mapping, non-finite
   reward rejection before clamp. The cortex adapter has none of these on
   `a0fd18e`; #131 adds a subset.
4. **Release cadence.** `neuromod` is a published crate; hybrid-fusion is
   pre-1.0 and iterating the contract. Keeping the adapter where the contract
   changes avoids lockstep releases.

Tension to record: AGENTS.md lists "Neuromodulator mapping / critic signals" as
owned by `limbic-critic`. The adapter's field mapping is *wire translation*
between two structs, not critic logic; limbic-critic still owns how a
`NeuroModulators` value is produced from objectives. If that reading is wrong,
the mapping function moves, not the adapter.

## 5. Hooks

### 5.1 `ResultTrack` (hybrid-fusion#62)

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ResultTrack { Ann, SnnNative, Hybrid }   // wire: "ann" | "snn-native" | "hybrid"
```

Where the step contract fixes the track:

- `HybridOutput` and `SpikingRouteOutput` carry `track: ResultTrack`, always
  `Hybrid`, set by the host, never by the backend.
- A bare `SpikingNetwork` driven by `HybridNetwork::forward` with the identity
  projector and no ANN input is still `Hybrid` by construction; a native-SNN
  track (`SnnNative`) can only be emitted by a host that never calls a
  `Transformer`. hybrid-fusion does not ship such a host today; `SnnNative` is
  reserved for Spikenaut-SNN / brainstem-daemon emitters (out of scope here).
- `ExpertRouteOutput` from a plain `ExpertRouter::route` is `Ann`; wrapped by
  `SpikingExpertRouter` it becomes part of a `Hybrid` output. The dense
  `ExpertRouteOutput` inside `SpikingRouteOutput.route` is *not* separately
  labelled `Ann`; it is a component of a hybrid result.
- `telemetry.rs` records gain a required `track` field; combining two records
  of different tracks is an explicit `HybridError::TrackMismatch { left, right }`
  (variant to be added under #62, not here).

### 5.2 `SpikeEncoding` / `SpikeDecoding` on `PortSpec`

```rust
pub enum SpikeEncodingKind { Rate, Latency, Delta, Population, Predictive, Other(String) }
pub struct SpikeEncoding { pub kind: SpikeEncodingKind, pub timesteps: usize, pub range: Option<(f32, f32)> }

pub enum SpikeDecodingKind { Count, Rate, FirstSpikeLatency, Membrane, Other(String) }
pub struct SpikeDecoding { pub kind: SpikeDecodingKind, pub window: usize }

pub struct PortSpec {
    pub dtype: Option<Dtype>,
    pub dims: Vec<DimSpec>,
    pub encoding: Option<SpikeEncoding>,   // new; Some only on Adapter→Snn edges
    pub decoding: Option<SpikeDecoding>,   // new; Some only on Snn→Adapter edges
}
```

Compatibility rule in `PortSpec::compatible_with` (`plan.rs:~195`): an `Snn`
stage input port with `encoding: Some(e)` and a producer with `encoding: None`
is a plan error ("stimuli reach the SNN with no declared encoding"); mismatched
`kind` or `timesteps` is a plan error; `range` mismatch is a warning in the
`NegotiationReport`, not an error, because the projector already bounds stimuli
(`hybrid.rs:35`). `SpikeEncoder::encoding()` / `SpikeDecoder::decoding()` are the
source of truth the plan compiler copies onto the port. `timesteps` is the
contract hook for multi-tick encodings; every encoder in this ADR reports `1`.

## 6. Where each fix in cortex-tensor PRs #130 and #131 lands

| PR / issue | Fix in cortex | Lands in hybrid-fusion as |
|---|---|---|
| #130 / #93 (1) | routing score `gate + spike_boost * activity` instead of `gate * (1 + spikes)`; `with_spike_boost` validated finite ≥ 0; `DEFAULT_SPIKE_BOOST = 1.0` | `SpikingExpertRouter::with_spike_boost`, same formula and saturation rule (`f32::MAX`, ±Inf gate keeps its value). **Caveat:** the boost is additive on gate *logits*; `ExpertRouter::route` returns *normalized* weights (`traits.rs:304-341`). See open question 8.1. |
| #130 / #93 (2) | `hidden = embedding * Σ_sel expert_weights[e] * activity[e] / max(activity)`, bounded by selected softmax mass; silent tick → dense selected mass | `SpikingRouteOutput.hidden_gain` computed with the same formula; the host multiplies, so the gain is also observable in telemetry |
| #130 / #93 (3) | post-step validation: out-of-range spike → `SnnSpikeOutOfRange`, duplicate → `SnnDuplicateSpike`, wrong-length readout → `SnnReadoutLength`, negative/Inf readout → `SnnInvalidReadout` | `HybridError` variants per §2.2; wrong-length readout uses the existing `InputLengthMismatch`. `HybridNetwork::forward` (`hybrid.rs:173`) switches from `SnnStep(text)` to `SnnSpikeOutOfRange` and gains the duplicate check |
| #131 / #91 | require one LIF per channel (Izhikevich bank auxiliary); `new` seeds identity wiring (neuron *i* ← channel *i* at max weight); `from_network` rejects a LIF neuron with no drivable weight | `NeuromodSnn` already separates `num_neurons` from `num_channels` and counts only LIF neurons (`neuromod_adapter.rs:454-456`). Port: (a) a drivable-weight check in `from_network` (currently `pub(crate)`, `:297`), (b) a new constructor `NeuromodSnn::identity_wired(channels, seed)` for the expert-*i*-at-neuron-*i* router case, since `with_seed`'s hashed dense weights (`:125-141`) do not give channel isolation. #131's "each channel driven alone spikes only its own neuron" test ports against that constructor |
| #131 / #92 (frozen) | `step_frozen` → `neuromod::SpikingNetwork::step_frozen`; `frozen_evaluation: true` | `NeuromodSnn::step_frozen` → `step_frozen_with_rng(.., &mut self.rng)`; advertise `RequiredFeature::new("frozen_evaluation")`. The bit-for-bit "weights, eligibility, thresholds, decay, modulators, STDP config unchanged" test ports as-is |
| #131 / #92 (RNG) | `with_seed` / `set_seed(Option)` / `seed()`; `caller_rng` true only while a seed is installed; `reset` reseeds | `NeuromodSnn` always owns a seeded RNG (`:173, :304`), so `caller_rng` is already truthful. `reset` semantics differ: hybrid-fusion restores weights and does **not** reseed (`:327-351`); cortex #131 reseeds and does **not** restore weights. Keep hybrid-fusion's (replay from original connectivity is the stronger guarantee) and port the mixed `step`/`step_frozen` replay test. See open question 8.3 |

Nothing from #130/#131 lands in cortex-tensor after the move.

## 7. Migration sequence

Versions: hybrid-fusion is `0.3.0` (unreleased v0.3.0 track, `CHANGELOG.md`);
cortex-tensor is `0.1.0`. Both are pre-1.0, so minor bumps carry breaking changes;
commit subjects use `!` per both AGENTS.md files.

1. **hybrid-fusion, additive (0.3.x, no `!`).** Land `SpikeEncoder`,
   `SpikeDecoder`, `RateEncoder`, `SpikeCountDecoder`, `SpikingNetwork::step_frozen`
   (defaulted), the new `HybridError` variants, `SpikingExpertRouter`, and the
   ported `src/snn/tests.rs` cases. Port #130 into `SpikingExpertRouter` and #131
   into `NeuromodSnn` in the same series. `PortSpec.encoding/decoding` as
   `Option` fields with `#[serde(default)]` so existing plans deserialize.
   `ResultTrack` lands under #62 on this same track.
2. **cortex-tensor, deprecate (0.1.x).** `#[deprecated(note = "moved to hybrid-fusion; see ADR 0001")]`
   on every `pub` item in `src/snn/`; README rows updated. The `neuromod` and
   `axon-encoder` features stay for one release so downstream builds do not break
   silently.
3. **cortex-tensor, remove (0.2.0, `snn!:`, issue #137).** Delete `src/snn/`,
   both features and dependencies, the four `CortexError::Snn*` variants, and
   `lib.rs:19,31`. Keep `HiddenStateBuffer` and `export_hidden`/`import_hidden`.
4. **hybrid-fusion `smoke/cortex-neuromod`.** Re-point at `SpikingExpertRouter`
   over a cortex `MoeRouter` (see 8.2 for how `MoeRouter` becomes an
   `ExpertRouter`). This is the exact-artifact gate that the relocated router
   reproduces the pre-move `selected_experts` for the #130 fixtures.
5. **Open PRs.**
   - cortex-tensor #130: **close without merging** once step 1 has landed the
     port with its seven tests; link this ADR. Rebasing it onto code that step 3
     deletes has no value.
   - cortex-tensor #131: **close without merging** once `NeuromodSnn` carries the
     #91/#92 behaviours and their tests. Its README row change is superseded by
     step 2.
   - cortex-tensor #133 (`axon-encoder` 0.5 → 0.6 bump): becomes moot at step 3;
     close then.
   - cortex-tensor #85 (audit tracker): #91, #92, #93 close via the hybrid-fusion
     PRs, with a comment naming the relocated tests.
   - hybrid-fusion #60 (CLAUDE.md): unaffected.

Until step 1 lands, #130 and #131 stay on hold as issue #61 already states.

## 8. Consequences and open questions

### Consequences

- One contract, one adapter, one error type for the boundary. The #91–#93 fixes
  are ported once.
- cortex-tensor loses its only hybrid path; it becomes strictly the `Ann` track.
  That is the point: ANN and hybrid results can no longer come out of the same
  `MoeOutput`.
- hybrid-fusion's default build stays trait-only. The `neuromod` feature remains
  the single tracked dependency exception (#44).
- `SpikeDecoder` sized by `num_neurons` is a semantic change from cortex; any
  caller that relied on `channels == neurons` must say so via
  `SpikingExpertRouter::new`'s `num_neurons >= num_experts` check.
- Spike validation moves into the host, so *every* `SpikingNetwork`, not only the
  router path, is checked for out-of-range and duplicate fired IDs.

### Open questions (not decided here)

1. **Logits vs normalized weights for the spike boost (#130 item 1).**
   `ExpertRouter::route` returns normalized weights; the additive boost is
   defined on pre-softmax gate logits. Options: (a) add
   `fn gate_logits(&mut self, embedding) -> Result<Vec<f32>>` to `ExpertRouter`
   with a default that errors, and have `SpikingExpertRouter` require it;
   (b) boost in log-space on the normalized weights and renormalize. (a)
   preserves #130's exact numbers; (b) keeps `ExpertRouter` unchanged. Needs a
   decision before step 1 ships `SpikingExpertRouter`.
2. **Who implements `ExpertRouter` for cortex's `MoeRouter`.** cortex-tensor must
   not depend on hybrid-fusion (`adapter.rs:118-120` says so explicitly), and
   hybrid-fusion must not depend on cortex-tensor. Candidates: the
   `smoke/cortex-neuromod` crate, or a tiny glue crate. Undecided.
3. **`reset` semantics for `NeuromodSnn`.** hybrid-fusion restores initial
   weights/thresholds and keeps the RNG stream; cortex #131 reseeds and keeps
   learned weights. This ADR keeps hybrid-fusion's behaviour; whether a
   `reset_and_reseed` convenience is wanted is open.
4. **`AxonEncoder`.** Dropping it from cortex leaves no `axon-encoder` adapter
   anywhere in the three repos. Re-homing it in hybrid-fusion needs a second
   dependency exception (a #44-style issue) or a home in `axon-encoder` itself,
   which would then depend on hybrid-fusion for `SpikeEncoder`. Not decided.
5. **limbic-critic ownership of the modulator mapping.** See §4. If the
   `NeuroModulators` → `neuromod::NeuroModulators` table is judged to be critic
   logic, it moves to limbic-critic and `NeuromodSnn` takes a pre-mapped value.
6. **`SnnNative` emitters.** No code in these three repos can emit `SnnNative`.
   Whether hybrid-fusion should ship a `SpikingNetwork`-only host that emits it,
   or leave that to Spikenaut-SNN / brainstem-daemon, is for #62.
7. **Multi-timestep encodings.** `SpikeEncoding.timesteps > 1` is declared but
   no host in this ADR steps more than one tick per forward. The contract for
   who loops (encoder, host, or backend) is undecided.
8. **cortex-tensor version for the removal.** `0.2.0` is assumed above; if the
   maintainers prefer to batch it with other `!` changes queued in #85
   (e.g. #126), the number moves but the sequence does not.
