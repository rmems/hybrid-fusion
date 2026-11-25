# ADR 0002: two SNN step contracts, one documented mapping, one parity gate

- **Status:** Accepted (2026-10-26)
- **Tracking:** Linear [RM-2360](https://linear.app/rpd-34/issue/RM-2360/snn-keep-cortex-tensors-snn-step-contract-in-sync-with-hybrid-fusions).
  Supersedes the "single owner / cortex-hybrid removal" part of
  [ADR 0001](0001-snn-step-contract.md) (§2 decision, §3, §4 last paragraph
  re cortex removal, §7 steps 2–5, and the "single ANN↔SNN step contract"
  wording throughout). Related: [RM-2205](https://linear.app/rpd-34/issue/RM-2205)
  (ADR 0001), [RM-2206](https://linear.app/rpd-34/issue/RM-2206) (closed not
  planned — cortex-hybrid keeps its SNN code), [RM-2208](https://linear.app/rpd-34/issue/RM-2208)
  (smoke dependency moved to crates.io here).
- **Trees cited:** hybrid-fusion `ea230f7`, cortex-hybrid `74d63ee` at ADR
  acceptance. The smoke lockfile now pins published cortex-hybrid 0.2.0
  (package VCS commit `704c3128f9bca71266fb5f40d41faa7862412c3c`).

## 1. Context

ADR 0001 planned for hybrid-fusion to become the *single* owner of the ANN↔SNN
step contract and for cortex-hybrid to lose `src/snn`. That removal did not
happen: on 2026-10-25 Raul decided cortex-hybrid **keeps its SNN code**
because the crate is meant to handle both ANN and SNN (Linear #137 /
RM-2206, closed not planned; the removal PRs
[rmems/cortex-hybrid#137](https://github.com/rmems/cortex-hybrid/pull/137)
and [#141](https://github.com/rmems/cortex-hybrid/pull/141) were closed
unmerged). Meanwhile hybrid-fusion PR
[#65](https://github.com/rmems/hybrid-fusion/pull/65) (`3c9acf5`) had already
landed hybrid-fusion's own copy of the contract, porting the cortex-hybrid
#91/#92/#93 fixes with some deliberate behavior changes.

The result is two parallel trait sets that can drift silently:

| cortex-hybrid `src/snn` | hybrid-fusion | Notes |
|---|---|---|
| `SnnBackend` | `SpikingNetwork` (`src/traits.rs`) | incl. `step_frozen` + `frozen_evaluation` on both |
| `SnnEncoder` / `SnnDecoder` | `SpikeEncoder` / `SpikeDecoder` (`src/encoding.rs`) | |
| `SpikingMoeRouter` | `SpikingExpertRouter` (`src/spiking_router.rs`) | |
| `src/snn/neuromod_adapter.rs` (`NeuromodNetwork`) | `src/neuromod_adapter.rs` (`NeuromodSnn`) | both wrap `neuromod` 0.7 |

## 2. Decision

**Each crate keeps its own traits.** cortex-hybrid does not implement or
depend on hybrid-fusion's traits, and hybrid-fusion does not re-export or
depend on cortex-hybrid's. The dependency direction forbids it — cortex-hybrid
must not depend on the orchestrator, and hybrid-fusion's public API must stay
free of concrete backend crates — and the two contracts serve different
callers:

- cortex-hybrid's `src/snn` is the **ANN+SNN reference path inside one crate**:
  `SpikingMoeRouter` composes its concrete `MoeRouter` and returns a
  `MoeOutput` (a scaled `Vec<f32>` hidden state sized by `EMBEDDING_DIM`).
- hybrid-fusion's contract is the **orchestrator's pluggable boundary**:
  generic over any `ExpertRouter` + `SpikingNetwork`, with capability
  negotiation, plan-port metadata, telemetry, and its own error vocabulary.

Drift is prevented mechanically, not by prose:

1. This ADR is the **mapping of record** (§3, §4). It is mirrored in
   cortex-hybrid as `docs/snn-contract-mapping.md`; when either contract
   changes, both documents change in the same PR series.
2. Cross-crate **parity tests** in
   `smoke/cortex-neuromod/tests/snn_contract_parity.rs` (CI: the smoke
   package runs in both the `validate` and `msrv` jobs) run the same seeded
   inputs through both trait sets. Agreement is asserted exactly where the
   contracts must agree, and every intentional difference is asserted, so a
   difference can only enter through an edited test that names it. **Scope
   of the gate:** the smoke package lockfile pins published cortex-hybrid
   0.2.0 (package VCS commit `704c3128f9bca71266fb5f40d41faa7862412c3c`),
   so the CI gate proves agreement with *that package*. A later cortex-hybrid
   change to a shared contract is invisible to the gate until the locked
   version moves — a change to either crate's side of the contract must
   therefore update the smoke lockfile (and extend these tests) in the same
   PR series; a green gate against a stale lockfile only proves agreement
   with the locked package.
3. Published cortex-hybrid 0.2.0 is past the #91/#92/#93 fix series and past
   RM-2364, so the parity tests compare like with like.

## 3. Trait-set mapping

| cortex-hybrid | hybrid-fusion | Mapping |
|---|---|---|
| `SnnBackend::channels()` | `SpikingNetwork::num_channels()` | stimulus width |
| — (implied: one neuron per channel) | `SpikingNetwork::num_neurons()` | output population; **separate axis in hybrid-fusion** (see §4.1) |
| `SnnBackend::step(&[f32])` | `SpikingNetwork::step(&[f32], &NeuroModulators)` | modulators are adapter state in cortex (`NeuromodNetwork.modulators`, `neuromod::NeuroModulators::default()`), an explicit per-call argument in hybrid-fusion (mapped per-field: `aux_dopamine` folded into `dopamine` clamped to `[0, 1]`, `cortisol → norepinephrine`, `acetylcholine` direct, `tempo` dropped, `serotonin` pinned to `0.0`) |
| `SnnBackend::step_frozen` (required; backends without a frozen path may delegate to `step` and report `frozen_evaluation = false`) | `SpikingNetwork::step_frozen` (default **fails closed**; overriders must advertise the `frozen_evaluation` feature) | hybrid-fusion's `SpikingExpertRouter::forward_frozen` gates on the capability before the encoder advances; cortex-hybrid's `SpikingMoeRouter::forward_frozen` relies on the trait contract |
| `SnnBackend::reset()` (infallible) | `SpikingNetwork::reset() -> Result<()>` (default: unsupported error) | both reset dynamics for a new epoch; RNG/weight semantics differ per adapter (§4.3) |
| `SnnCapabilities { backend_name, frozen_evaluation, plasticity, caller_rng, neuromodulation }` | `BackendCapabilities::snn(..)` + `RequiredFeature` strings `step`, `frozen_evaluation`, `plasticity`, `caller_rng`, `neuromodulation` | two vocabularies for the same facts; the adapters report identical values (asserted by the parity tests) |
| `SnnStepOutput { spikes: Vec<usize> }` | `Vec<usize>` fired IDs | same index space (`0..` output population), no wrapper in hybrid-fusion |
| `SnnEncoder::encode(&ann_signal, channels)`, `reset()` | `SpikeEncoder::encode`, `reset()`, plus `encoding()` (`SpikeEncoding` metadata) | identical semantics; hybrid-fusion adds declared plan-port metadata |
| `SnnDecoder::decode(&SnnStepOutput, channels) -> Result<Vec<f32>>` (fallible above `MAX_SNN_CHANNELS`) | `SpikeDecoder::decode(&[usize], num_neurons) -> Result<Vec<f32>>`, plus `decoding()` metadata | **sized by channel count vs output population**, with different standalone out-of-range spike behavior (§4.2) |
| `RateEncoder { gain }` | `RateEncoder { gain }` | identical tiling semantics |
| `SpikeCountDecoder` | `SpikeCountDecoder` | counts per channel / per neuron; out-of-range IDs silently dropped vs rejected (§4.2) |
| `SpikingMoeRouter<B, E, D>` over the concrete `MoeRouter` | `SpikingExpertRouter<R, S, E, D>` over any `ExpertRouter` | same pipeline: gate scores → encode → step → decode → additive spike boost → top-k |
| `DEFAULT_SPIKE_BOOST = 1.0`, `with_spike_boost` (finite, ≥ 0) | `DEFAULT_SPIKE_BOOST = 1.0`, `with_spike_boost` (finite, ≥ 0) | identical |
| `MoeOutput { expert_weights, selected_experts, hidden: Vec<f32> }` | `SpikingRouteOutput { route: ExpertRouteOutput, fired_neurons, num_neurons, hidden_gain: f32 }` | same boosted softmax weights and selection; hidden state vs scalar gain (§4.4), plus entropy + fired IDs on the hybrid-fusion side |
| `NeuromodNetwork` (`new(num_lif, num_izh, num_channels)`, `from_network`, `with_seed`) | `NeuromodSnn` (`with_seed(channels, neurons, seed)`, `identity_wired(channels, seed)`, `reseed`) | both wrap `neuromod::SpikingNetwork`; wiring and reset semantics differ (§4.3, §4.5) |
| `src/snn/axon_adapter.rs` (`AxonEncoder<E: SnnEncoder-facing axon_encoder::Encoder>`, feature `axon-encoder`) | `src/axon_adapter.rs` (`AxonEncoderAdapter<E: axon_encoder::Encoder>` as `DomainAdapter`, feature `axon-encoder`) | same bridge — one `encode_step` per call, `±1.0` per spike into a per-channel stimulus, `reset` delegates. Layer differences: the cortex side takes a ready `ann_signal`, while the hybrid side's `DomainAdapter` input is a hidden `Tensor` it pools, resizes, and `tanh`s first (the parity test feeds the tanh'd signal so both steps see identical input); the hybrid side nets contributions then clamps at `±STIMULUS_BOUND` (identical while the net stays in bounds), snapshots encoder state for failure atomicity, and, as hybrid-only policy, rejects every `PopulationEncoder` (unseeded RNG), multi-tick `time_model`s, and fixed-channel-count mismatches. The cortex side checks neither `time_model` nor configured channel width: at encode time it rejects inputs above the `u16` spike-id space, a `PopulationEncoder` only when `inputs × neurons` overflows that space, and out-of-range spike channels (`SnnSpikeOutOfRange`) |
| `CortexError::{SnnBackend, SnnChannelMismatch, SnnNan, SnnInvalidStimulus, SnnSpikeOutOfRange, SnnDuplicateSpike, InputLengthMismatch, SnnReadoutLength, SnnInvalidReadout}` | `HybridError::{SnnStep, ConfigMismatch, NonFinite, SnnSpikeOutOfRange, SnnDuplicateSpike, InputLengthMismatch, SnnInvalidReadout}` | same failure conditions except the ±Inf stimulus policy (§4.6); both hosts reject NaN before the tick and validate spikes/readouts after it. `neuromod::StepError` mapping: cortex-hybrid maps `InputLenMismatch` → `InputLengthMismatch` and `NonFiniteStimulus` → `SnnInvalidStimulus`, the rest stay `SnnBackend` (RM-2364); hybrid-fusion maps `InputLenMismatch` → `InputLengthMismatch` and every other variant, `NonFiniteStimulus` included, to `SnnStep` |

## 4. Intentional differences

Each difference below is **asserted by the parity tests where it is
externally observable** through both crates' public APIs. The one exception
is the hybrid-fusion half of §4.5: its rejection path is `pub(crate)` and
its public constructors cannot produce a dead network, so the parity tests
assert cortex-hybrid's public rejection and hybrid-fusion's in-crate tests
cover its own side; this ADR documents the boundary.

### 4.1 Width axes

cortex-hybrid: `channels()` is the whole width contract — the adapter requires
**one LIF neuron per channel** (`NeuromodNetwork::new(8, 1, 4)` is an
`InvalidConfig` error) because `neuromod` reports LIF spikes only.
hybrid-fusion keeps stimulus width and output population separate:
`NeuromodSnn::with_seed(4, 8, ..)` is fine, `num_neurons() >= num_experts()`
is the router's requirement, and `identity_wired` is the one-neuron-per-channel
special case.

### 4.2 Decoder fallibility and sizing

cortex-hybrid's `SnnDecoder` is sized by the channel count. It is fallible
when the count exceeds `MAX_SNN_CHANNELS` (1,048,576 channels), the boundary's
allocation cap; `SpikingMoeRouter` enforces the same cap before stepping. The
`SpikeCountDecoder` silently drops out-of-range IDs, and the
`SpikingMoeRouter` host validates the spike list before decoding, so a
compliant flow never reaches the decoder with a bad ID. hybrid-fusion's
`SpikeDecoder` is fallible, sized by `num_neurons` (backends with
`num_neurons != num_channels` keep working), and its `SpikeCountDecoder`
rejects out-of-range fired IDs with `HybridError::SnnSpikeOutOfRange`:
fail closed rather than hide a backend contract violation. Both **hosts**
reject out-of-range and duplicate fired IDs with equivalent typed errors;
only standalone-decoder behavior differs.

### 4.3 Reset and replay semantics

- cortex-hybrid `NeuromodNetwork::reset`: rewinds the replay RNG to its seed
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

cortex-hybrid's `NeuromodNetwork::new` identity-wires LIF neuron *i* to
channel *i* at the configured maximum weight, takes an explicit Izhikevich
bank size, and exposes `from_network(SpikingNetwork)` publicly, which is also
where the #91 never-spiking-neuron rejection is observable. hybrid-fusion's
public constructors (`with_seed`, `identity_wired`) always seed strictly
positive weights, so the #91 rejection lives behind the `pub(crate)`
`from_network` — no `neuromod` type may cross hybrid-fusion's public API.
hybrid-fusion's adapter additionally enforces a 64 MiB matrix budget and maps
`neuromod::StepError` exhaustively (`InputLenMismatch` → `InputLengthMismatch`,
every other variant → `SnnStep`); cortex-hybrid's adapter maps the two
structural cases (`InputLenMismatch` → `InputLengthMismatch`,
`NonFiniteStimulus` → `SnnInvalidStimulus`) and keeps the rest in `SnnBackend`
(RM-2364). Coverage note: the parity tests assert
cortex-hybrid's public rejection (`from_network` on a blank network);
hybrid-fusion's `pub(crate)` rejection path and its matrix budget are covered
by that crate's in-crate tests, not by the cross-crate gate, because neither
is reachable through hybrid-fusion's public API.

### 4.6 ±Inf stimulus policy

cortex-hybrid's `SpikingMoeRouter::forward` / `forward_frozen` reject a ±Inf
gate score, or a ±Inf encoded stimulus sample, with
`CortexError::SnnInvalidStimulus` **before** the encoder or backend runs;
NaN keeps `NanRoutingScore` / `SnnNan` (RM-2364, cortex-hybrid #155).
hybrid-fusion's `SpikingExpertRouter` ranks ±Inf gate logits — its
finite-value policy rejects only NaN pre-tick, and the additive spike boost
keeps ±Inf gate values — and passes the encoded ±Inf stimulus to the
backend, whose own failure-atomic preflight rejects it
(`neuromod::StepError::NonFiniteStimulus` → `HybridError::SnnStep`). Both
hosts fail closed on ±Inf; they differ in the rejecting layer (router
preflight in cortex-hybrid, backend preflight here), in the error variant,
and in whether the encoder consumes a tick before the rejection (observable
through a stateful encoder; the shared `RateEncoder` is stateless). The
parity tests assert all three, using a mirrored stateful phase encoder and
the post-failure recovery tick.

## 5. Fix parity (#91–#93)

All four fixes exist on **both** sides; the parity tests keep it that way:

| Fix | cortex-hybrid | hybrid-fusion |
|---|---|---|
| #91 never-spiking channel rejection + drivable wiring | `NeuromodNetwork::checked` / `new` (PR #131), public via `from_network` | `NeuromodSnn::from_network` (`pub(crate)`) + seeded constructors (PR #65) |
| #92 real frozen step + truthful `caller_rng` | `step_frozen_with_rng`, `with_seed`/`set_seed`, `caller_rng` reflects the installed seed (PR #131) | `step_frozen_with_rng`, always-owned seeded RNG, `frozen_evaluation` + `caller_rng` features (PR #65) |
| #92-style held-out never runs the learning path | trait contract: backends without a frozen path may delegate | trait default fails closed + host capability gate |
| #93 additive spike boost on gate logits (`gate + spike_boost × activity`, saturating at `f32::MAX`, `±Inf` gates keep their value — though RM-2364 now rejects a ±Inf *gate score* in cortex-hybrid's router before the boost runs, §4.6) | `SpikingMoeRouter` (PR #130/#131 series) | `SpikingExpertRouter` (PR #65) |
| #93 bounded hidden gain (activity normalized to the tick peak, silent-tick fallback) | same formula | same formula |

## 6. Consequences

- No document, module doc, or changelog in either crate may claim that
  hybrid-fusion is the *single* owner of the step contract or that
  cortex-hybrid is ANN-only. The stale claims from PR #65 (AGENTS.md,
  REVIEW.md, CHANGELOG, `docs/implementing-backends.md`, `src/traits.rs`,
  `src/encoding.rs`, `src/spiking_router.rs`) are corrected in the same
  change that adds this ADR; ADR 0001 keeps its history with a superseded
  marker.
- A change to one contract that should apply to both must now say so: edit
  both crates, both mapping documents, extend the parity tests, and **update
  the smoke lockfile** when cortex-hybrid's behavior changes — the gate only
  tests the locked package — or explicitly record a new intentional difference
  here.
- The `smoke/cortex-neuromod` dependency (RM-2208) is published cortex-hybrid
  0.2.0, locked exactly in `Cargo.lock` — past the #91–#93 fix series and past
  RM-2364, whose contract changes are folded into §3, §4.6, and the parity
  tests here. Future package updates must keep the parity tests green, and a
  green gate against a stale lockfile only proves agreement with the locked
  package, not with cortex-hybrid `main`.
- `AxonEncoder` (cortex-hybrid, `axon-encoder` feature) now has a
  hybrid-fusion counterpart, `AxonEncoderAdapter` (`axon-encoder` feature),
  added with RM-1811. §3 carries the row and
  `smoke/cortex-neuromod/tests/snn_contract_parity.rs` replays identical
  `EmbeddingRateEncoder` instances through both adapters, pinning the shared
  `encode_step` → `±1.0`-per-channel bridge across streaming ticks and
  `reset`.
