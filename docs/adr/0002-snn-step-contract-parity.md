# ADR 0002: two SNN step contracts, one documented mapping, one parity gate

- **Status:** Accepted (2026-10-26)
- **Tracking:** Linear [RM-2360](https://linear.app/rpd-34/issue/RM-2360/snn-keep-cortex-tensors-snn-step-contract-in-sync-with-hybrid-fusions).
  Supersedes the "single owner / cortex-tensor removal" part of
  [ADR 0001](0001-snn-step-contract.md) (§2 decision, §3, §4 last paragraph
  re cortex removal, §7 steps 2–5, and the "single ANN↔SNN step contract"
  wording throughout). Related: [RM-2205](https://linear.app/rpd-34/issue/RM-2205)
  (ADR 0001), [RM-2206](https://linear.app/rpd-34/issue/RM-2206) (closed not
  planned — cortex-tensor keeps its SNN code), [RM-2208](https://linear.app/rpd-34/issue/RM-2208)
  (smoke pin bump, unblocked here).
- **Trees cited:** hybrid-fusion `ea230f7`, cortex-tensor `f591f2e`
  (the smoke pin this ADR lands with).

## 1. Context

ADR 0001 planned for hybrid-fusion to become the *single* owner of the ANN↔SNN
step contract and for cortex-tensor to lose `src/snn`. That removal did not
happen: on 2026-10-25 Raul decided cortex-tensor **keeps its SNN code**
because the crate is meant to handle both ANN and SNN (Linear #137 /
RM-2206, closed not planned; the removal PRs
[rmems/cortex-tensor#137](https://github.com/rmems/cortex-tensor/pull/137)
and [#141](https://github.com/rmems/cortex-tensor/pull/141) were closed
unmerged). Meanwhile hybrid-fusion PR
[#65](https://github.com/rmems/hybrid-fusion/pull/65) (`3c9acf5`) had already
landed hybrid-fusion's own copy of the contract, porting the cortex-tensor
#91/#92/#93 fixes with some deliberate behavior changes.

The result is two parallel trait sets that can drift silently:

| cortex-tensor `src/snn` | hybrid-fusion | Notes |
|---|---|---|
| `SnnBackend` | `SpikingNetwork` (`src/traits.rs`) | incl. `step_frozen` + `frozen_evaluation` on both |
| `SnnEncoder` / `SnnDecoder` | `SpikeEncoder` / `SpikeDecoder` (`src/encoding.rs`) | |
| `SpikingMoeRouter` | `SpikingExpertRouter` (`src/spiking_router.rs`) | |
| `src/snn/neuromod_adapter.rs` (`NeuromodNetwork`) | `src/neuromod_adapter.rs` (`NeuromodSnn`) | both wrap `neuromod` 0.7 |

## 2. Decision

**Each crate keeps its own traits.** cortex-tensor does not implement or
depend on hybrid-fusion's traits, and hybrid-fusion does not re-export or
depend on cortex-tensor's. The dependency direction forbids it — cortex-tensor
must not depend on the orchestrator, and hybrid-fusion's public API must stay
free of concrete backend crates — and the two contracts serve different
callers:

- cortex-tensor's `src/snn` is the **ANN+SNN reference path inside one crate**:
  `SpikingMoeRouter` composes its concrete `MoeRouter` and returns a
  `MoeOutput` (a scaled `Vec<f32>` hidden state sized by `EMBEDDING_DIM`).
- hybrid-fusion's contract is the **orchestrator's pluggable boundary**:
  generic over any `ExpertRouter` + `SpikingNetwork`, with capability
  negotiation, plan-port metadata, telemetry, and its own error vocabulary.

Drift is prevented mechanically, not by prose:

1. This ADR is the **mapping of record** (§3, §4). It is mirrored in
   cortex-tensor as `docs/snn-contract-mapping.md`; when either contract
   changes, both documents change in the same PR series.
2. Cross-crate **parity tests** in
   `smoke/cortex-neuromod/tests/snn_contract_parity.rs` (CI: the smoke
   package runs in both the `validate` and `msrv` jobs) run the same seeded
   inputs through both trait sets. Agreement is asserted exactly where the
   contracts must agree, and every intentional difference is asserted, so a
   difference can only enter through an edited test that names it.
3. The smoke package's cortex-tensor pin is bumped past the #91/#92/#93 fix
   series (`f591f2e`, RM-2208), so the parity tests compare like with like.

## 3. Trait-set mapping

| cortex-tensor | hybrid-fusion | Mapping |
|---|---|---|
| `SnnBackend::channels()` | `SpikingNetwork::num_channels()` | stimulus width |
| — (implied: one neuron per channel) | `SpikingNetwork::num_neurons()` | output population; **separate axis in hybrid-fusion** (see §4.1) |
| `SnnBackend::step(&[f32])` | `SpikingNetwork::step(&[f32], &NeuroModulators)` | modulators are adapter state in cortex (`NeuromodNetwork.modulators`, `neuromod::NeuroModulators::default()`), an explicit per-call argument in hybrid-fusion (mapped per-field: `aux_dopamine` folded into `dopamine` clamped to `[0, 1]`, `cortisol → norepinephrine`, `acetylcholine` direct, `tempo` dropped, `serotonin` pinned to `0.0`) |
| `SnnBackend::step_frozen` (required; backends without a frozen path may delegate to `step` and report `frozen_evaluation = false`) | `SpikingNetwork::step_frozen` (default **fails closed**; overriders must advertise the `frozen_evaluation` feature) | hybrid-fusion's `SpikingExpertRouter::forward_frozen` gates on the capability before the encoder advances; cortex-tensor's `SpikingMoeRouter::forward_frozen` relies on the trait contract |
| `SnnBackend::reset()` (infallible) | `SpikingNetwork::reset() -> Result<()>` (default: unsupported error) | both reset dynamics for a new epoch; RNG/weight semantics differ per adapter (§4.3) |
| `SnnCapabilities { backend_name, frozen_evaluation, plasticity, caller_rng, neuromodulation }` | `BackendCapabilities::snn(..)` + `RequiredFeature` strings `step`, `frozen_evaluation`, `plasticity`, `caller_rng`, `neuromodulation` | two vocabularies for the same facts; the adapters report identical values (asserted by the parity tests) |
| `SnnStepOutput { spikes: Vec<usize> }` | `Vec<usize>` fired IDs | same index space (`0..` output population), no wrapper in hybrid-fusion |
| `SnnEncoder::encode(&ann_signal, channels)`, `reset()` | `SpikeEncoder::encode`, `reset()`, plus `encoding()` (`SpikeEncoding` metadata) | identical semantics; hybrid-fusion adds declared plan-port metadata |
| `SnnDecoder::decode(&SnnStepOutput, channels) -> Vec<f32>` (infallible) | `SpikeDecoder::decode(&[usize], num_neurons) -> Result<Vec<f32>>`, plus `decoding()` metadata | **sized by channel count vs output population**, and infallible vs fail-closed (§4.2) |
| `RateEncoder { gain }` | `RateEncoder { gain }` | identical tiling semantics |
| `SpikeCountDecoder` | `SpikeCountDecoder` | counts per channel / per neuron; out-of-range IDs silently dropped vs rejected (§4.2) |
| `SpikingMoeRouter<B, E, D>` over the concrete `MoeRouter` | `SpikingExpertRouter<R, S, E, D>` over any `ExpertRouter` | same pipeline: gate scores → encode → step → decode → additive spike boost → top-k |
| `DEFAULT_SPIKE_BOOST = 1.0`, `with_spike_boost` (finite, ≥ 0) | `DEFAULT_SPIKE_BOOST = 1.0`, `with_spike_boost` (finite, ≥ 0) | identical |
| `MoeOutput { expert_weights, selected_experts, hidden: Vec<f32> }` | `SpikingRouteOutput { route: ExpertRouteOutput, fired_neurons, num_neurons, hidden_gain: f32 }` | same boosted softmax weights and selection; hidden state vs scalar gain (§4.4), plus entropy + fired IDs on the hybrid-fusion side |
| `NeuromodNetwork` (`new(num_lif, num_izh, num_channels)`, `from_network`, `with_seed`) | `NeuromodSnn` (`with_seed(channels, neurons, seed)`, `identity_wired(channels, seed)`, `reseed`) | both wrap `neuromod::SpikingNetwork`; wiring and reset semantics differ (§4.3, §4.5) |
| `src/snn/axon_adapter.rs` (`AxonEncoder` over `axon-encoder`, feature `axon-encoder`) | — | no hybrid-fusion counterpart; cortex-tensor-only adapter, out of this mapping's parity scope |
| `CortexError::{SnnBackend, SnnChannelMismatch, SnnNan, SnnSpikeOutOfRange, SnnDuplicateSpike, SnnReadoutLength, SnnInvalidReadout}` | `HybridError::{SnnStep, ConfigMismatch, NonFinite, SnnSpikeOutOfRange, SnnDuplicateSpike, InputLengthMismatch, SnnInvalidReadout}` | same failure conditions, different vocabularies; both hosts reject NaN before the tick and validate spikes/readouts after it |

## 4. Intentional differences (each asserted by the parity tests)

### 4.1 Width axes

cortex-tensor: `channels()` is the whole width contract — the adapter requires
**one LIF neuron per channel** (`NeuromodNetwork::new(8, 1, 4)` is an
`InvalidConfig` error) because `neuromod` reports LIF spikes only.
hybrid-fusion keeps stimulus width and output population separate:
`NeuromodSnn::with_seed(4, 8, ..)` is fine, `num_neurons() >= num_experts()`
is the router's requirement, and `identity_wired` is the one-neuron-per-channel
special case.

### 4.2 Decoder fallibility and sizing

cortex-tensor's `SnnDecoder` is infallible, sized by the channel count, and
its `SpikeCountDecoder` silently drops out-of-range IDs; the
`SpikingMoeRouter` host validates the spike list before decoding, so a
compliant flow never reaches the decoder with a bad ID. hybrid-fusion's
`SpikeDecoder` is fallible, sized by `num_neurons` (backends with
`num_neurons != num_channels` keep working), and its `SpikeCountDecoder`
rejects out-of-range fired IDs with `HybridError::SnnSpikeOutOfRange`:
fail closed rather than hide a backend contract violation. Both **hosts**
reject out-of-range and duplicate fired IDs with equivalent typed errors;
only standalone-decoder behavior differs.

### 4.3 Reset and replay semantics

- cortex-tensor `NeuromodNetwork::reset`: rewinds the replay RNG to its seed
  and calls neuromod's reset (which keeps learned weights). Replay after
  `reset` is exact for frozen-only stepping; after learning steps it depends
  on what R-STDP changed.
- hybrid-fusion `NeuromodSnn::reset`: restores the construction-time weights
  and thresholds (neuromod's reset leaves them as learned) and does **not**
  reseed the RNG; the documented replay recipe is `reset(); reseed(seed)`.
- The parity tests pin both recipes and additionally prove hybrid-fusion's
  bare `reset` continues the RNG stream bit-for-bit (it equals a
  never-reset instance's next run), i.e. the two crates' resets are
  observably different on purpose.

### 4.4 Output shape

`SpikingMoeRouter::forward` returns the scaled hidden state
(`embedding × spiking_mass`, a `Vec<f32>` of `EMBEDDING_DIM`). The host
scales. `SpikingExpertRouter::forward` returns the scalar `hidden_gain` in
`[0, 1]` (plus the fired IDs and the output population, for telemetry and
validation), and the caller scales. Both compute the identical bounded gain:
selected softmax mass weighted by activity normalized to the tick's peak,
falling back to the dense selected mass on a silent tick.

### 4.5 Adapter construction and the public boundary

cortex-tensor's `NeuromodNetwork::new` identity-wires LIF neuron *i* to
channel *i* at the configured maximum weight, takes an explicit Izhikevich
bank size, and exposes `from_network(SpikingNetwork)` publicly, which is also
where the #91 never-spiking-neuron rejection is observable. hybrid-fusion's
public constructors (`with_seed`, `identity_wired`) always seed strictly
positive weights, so the #91 rejection lives behind the `pub(crate)`
`from_network` — no `neuromod` type may cross hybrid-fusion's public API.
hybrid-fusion's adapter additionally enforces a 64 MiB matrix budget and an
exhaustive `StepError` mapping.

## 5. Fix parity (#91–#93)

All four fixes exist on **both** sides; the parity tests keep it that way:

| Fix | cortex-tensor | hybrid-fusion |
|---|---|---|
| #91 never-spiking channel rejection + drivable wiring | `NeuromodNetwork::checked` / `new` (PR #131), public via `from_network` | `NeuromodSnn::from_network` (`pub(crate)`) + seeded constructors (PR #65) |
| #92 real frozen step + truthful `caller_rng` | `step_frozen_with_rng`, `with_seed`/`set_seed`, `caller_rng` reflects the installed seed (PR #131) | `step_frozen_with_rng`, always-owned seeded RNG, `frozen_evaluation` + `caller_rng` features (PR #65) |
| #92-style held-out never runs the learning path | trait contract: backends without a frozen path may delegate | trait default fails closed + host capability gate |
| #93 additive spike boost on gate logits (`gate + spike_boost × activity`, saturating at `f32::MAX`, `±Inf` gates keep their value) | `SpikingMoeRouter` (PR #130/#131 series) | `SpikingExpertRouter` (PR #65) |
| #93 bounded hidden gain (activity normalized to the tick peak, silent-tick fallback) | same formula | same formula |

## 6. Consequences

- No document, module doc, or changelog in either crate may claim that
  hybrid-fusion is the *single* owner of the step contract or that
  cortex-tensor is ANN-only. The stale claims from PR #65 (AGENTS.md,
  REVIEW.md, CHANGELOG, `docs/implementing-backends.md`, `src/traits.rs`,
  `src/encoding.rs`, `src/spiking_router.rs`) are corrected in the same
  change that adds this ADR; ADR 0001 keeps its history with a superseded
  marker.
- A change to one contract that should apply to both must now say so: edit
  both crates, both mapping documents, and extend the parity tests, or
  explicitly record a new intentional difference here.
- The `smoke/cortex-neuromod` pin (RM-2208) is bumped to `f591f2e` in the
  same change; future pin bumps only need to keep the parity tests green.
- `AxonEncoder` (cortex-tensor, `axon-encoder` feature) has no
  hybrid-fusion counterpart and is out of parity scope; if it ever grows one,
  extend §3 and the parity tests in the same PR.
