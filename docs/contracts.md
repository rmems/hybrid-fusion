# Cross-crate contract: shapes, dtypes, finite values, result tracks

Linear RM-2211. Docs only. Describes what the code does today, not what it should do; unresolved items are in [Known gaps](#known-gaps).

Citations are `repo:path:line` at these commits. Re-verify against the pins before relying on a row.

| Alias | Repo | Commit |
|-------|------|--------|
| `ep` | rmems/engram-parser | `dbc46cf` |
| `ct` | rmems/cortex-tensor | `a0fd18e` |
| `hf` | rmems/hybrid-fusion | `506ff5b` |

## 1. Shape order

Rank-2 `[vocab, hidden]` example. Row-major everywhere except the raw GGUF directory.

| Layer | Order | Evidence |
|-------|-------|----------|
| GGUF `ne[]` / `Tensor::dims` | Innermost-first (`dims[0]` is the contiguous row width) | `ep:src/gguf/tensor.rs:465-466`; block check on `dims[0]` at `tensor.rs:369-390`; `dequant.rs:40` |
| engram `TensorInfo.shape` | Outermost-first. GGUF dims are reversed on the way in; `native_dims()` recovers the on-disk order | `ep:src/checkpoint/types.rs:100-111`, `123-128`, `141-146`; applied at `ep:src/checkpoint/gguf.rs:193`; Safetensors is already outermost-first (`types.rs:94-95`) |
| engram dequant inputs | Back to innermost-first: `dequantize_*` take GGUF-order `dims` | `ep:src/gguf/dequant.rs:40-41`, `94` |
| cortex binding | Outermost-first, taken verbatim from `info.shape.dims()`. Token embedding is `dims[0]=vocab`, `dims[1]=hidden` | `ct:src/moe/checkpoint.rs:63-68`, `165-171`, `309-317`; test `tensor_inventory_uses_outermost_first_shapes` at `:511-524` |
| cortex `Tensor` | Row-major, `data[i * ncols + j]` | `ct:src/tensor/mod.rs:12-22`, `64-67` |
| hf `Tensor` | Row-major `Vec<f32>` plus `shape`; every dim must be `> 0` | `hf:src/tensor.rs:5-28` |
| hf hidden state | `[seq, dim]` (rank 2) or `[dim]` (rank 1) | `hf:src/validate.rs:43-47`, `66-75` |
| hf `PortSpec.dims` | Per-axis `DimSpec` list, compared positionally by axis index. Empty means rank unconstrained, not scalar. Axis order is not stated in the type docs; it is positional only | `hf:src/plan.rs:144-157`, `162-170`, `208-219` |
| hf `TensorManifestEntry.shape` | Same `Vec<usize>` convention as `Tensor::shape`; zero dims and `[]` allowed here | `hf:src/types.rs:172-178`, `186-191` |

Net: engram, cortex, and hf agree on outermost-first row-major. The only innermost-first surfaces are the raw GGUF `Tensor::dims` and the engram dequant function arguments.

## 2. Dtype vocabulary

Three separate enums, no conversion code between them. hf has no engram dependency (`hf:src/topology.rs:186-188`) and no `From`/`TryFrom` exists. This table is a reading of the names, not an implemented mapping.

| engram `TensorDType` (`ep:src/checkpoint/types.rs:169-213`) | cortex `TensorKind` / binding (`ct:src/moe/checkpoint.rs:72-80`, `294-307`) | hf `types::Dtype` (`hf:src/types.rs:37-67`) |
|---|---|---|
| `F32` | `F32`; the only dtype the gate may have (`checkpoint.rs:147`) | `F32` |
| `F16` | `F16`; token rows decoded via `f16_bits_to_f32` (`:207-215`) | `F16` |
| `BF16` | `Other("BF16")`, rejected as unsupported (`:246`, `:305`) | `BF16` |
| `F64` | `Other("F64")`, rejected | `F64` |
| `I64` `I32` `I16` `I8` `U64` `U32` `U16` `U8` | `Other(label)`, rejected | same names, all present |
| `Bool` (`"BOOL"`) | `Other("BOOL")`, rejected | `Bool` |
| `F8E4M3`, `F8E5M2` | `Other`, rejected | `F8E4M3`, `F8E5M2` |
| `I4`, `U4` | `Other`, rejected | **unmapped** (no variant) |
| `GgufPacked` Q8_0 | `Q8Block`, decoded via `dequantize_q8_0` (`:216-224`) | **unmapped** (packed GGML types have no hf variant) |
| `GgufPacked` Q5_K / Q6_K | `Q5KBlock` / `Q6KBlock`, decoded (`:225-242`) | **unmapped** |
| `GgufPacked` IQ3_S | `Iq3S`, explicit error (`:243-245`); engram does not decode it (`ep:src/gguf/dequant.rs:94-102`, `129`) | **unmapped** |
| `GgufPacked` IQ3_M (internal id `0x4949334D`, not GGUF wire 31) | `Other`; no `TensorKind` | **unmapped** |
| `GgufPacked` all other (Q4_*, Q5_0/1, Q2_K, Q3_K, Q4_K, Q8_1, Q8_K, IQ*) | `Other`, rejected; engram fail-closes (`dequant.rs:104-`) | **unmapped** |
| `Other("F8_E8M0")`, `Other("C64")` (not normalized, `types.rs:211`) | `Other`, rejected | `F8E8M0`, `C64` exist but engram never produces them as typed values |
| (none) | (none) | `F4`, `F6E2M3`, `F6E3M2`, `F8E4M3Fnuz`, `F8E5M2Fnuz` have no engram counterpart |

How GGUF dtypes reach engram: only scalars normalize (`F32 F16 BF16 F64 I8 I16 I32 I64`); every block layout becomes `GgufPacked { ggml_type }` (`ep:src/checkpoint/gguf.rs:201-213`). Safetensors labels: `ep:src/checkpoint/safetensors.rs:179-200`.

Two other dtype sets exist and are not the same as the above:

| Set | Values | Evidence |
|-----|--------|----------|
| cortex stage `DType` | `F32`, `F16`, `BF16`; reference executor advertises `F32` only | `ct:src/stage.rs:56-85`, `1097` |
| hf `BackendCapabilities.dtypes` | `BTreeSet<hf Dtype>`; negotiation compares by enum equality | `hf:src/capabilities.rs:28-30`, `152`, `927-931` |

## 3. Finite-value policy (NaN, +Inf, -Inf)

| Crate / stage | NaN | +Inf / -Inf | Evidence |
|---|---|---|---|
| engram packed decode | Not inspected. Output is whatever the block bytes decode to | Not inspected. `f16_bits_to_f32` maps exp=31 to Inf/NaN bit patterns unchanged | `ep:src/gguf/tensor.rs:587-599`; no finite check in `ep:src/gguf/dequant.rs` |
| cortex checkpoint bind (`decode_f32`, F16/quant rows) | Not inspected; copied via `from_le_bytes` | Not inspected | `ct:src/moe/checkpoint.rs:356-375`, `207-215` |
| cortex softmax | Any NaN in row: whole row NaN | One `+Inf`: one-hot. k>1 `+Inf`: `1/k` split. All `-Inf`: uniform `1/n` | `ct:src/tensor/finite.rs:16-27` |
| cortex LayerNorm / RMSNorm | Non-finite in a row: that output row all NaN. `eps` non-finite or `<= 0`: `InvalidEpsilon` error | same | `finite.rs:31-43`, `116-123` |
| cortex L2 normalize | Any non-finite: fill with NaN | same | `finite.rs:45-52` |
| cortex activations (relu, gelu, silu, fast sigmoid) | NaN in, NaN out | `+Inf`: `+Inf` (sigmoid `1`); `-Inf`: `0` (sigmoid `-1`) | `finite.rs:54-66`; merged in cortex PR #124 (`bbb8c95`). Read from source at `a0fd18e`; not exercised by the hf smoke harness, which pins pre-#124 `5cbee81` (gap 12) |
| cortex `sum` / `mean` / `max_val` | Any NaN: NaN (`max_val` ranks with an explicit NaN check, not `f32::max`) | `+Inf` and `-Inf` together: NaN for sum/mean; one signed infinity: that infinity | `finite.rs:68-87` |
| hf forward path (embedding, stimuli) | Rejected: `HybridError::NonFinite { stage, index }` before `SpikingNetwork::step` | Rejected, same error | `hf:src/validate.rs:51-56`; `hf:src/hybrid.rs:157`, `160` |
| hf hidden state, MoE gate weights, modulators, routing scores | Rejected (NaN routing score: `InvalidConfig`; non-finite or negative expert weight: `InvalidConfig`; non-finite dopamine: `SnnStep`) | `routing.rs`: `+-Inf` scores are allowed, NaN is not | `hf:src/validate.rs:203-231` (tests); `hf:src/routing.rs:60-77`; `hf:src/reverse.rs:174-182`; `hf:src/neuromod_adapter.rs:371-396` |
| hf serde of `HybridOutput` / `Tensor` | Serializes as JSON `null` | Serializes as JSON `null`, sign lost | `hf:src/types.rs:297-332`, `hf:src/tensor.rs:5-9` (plain derives, no custom attrs) |

Verified for RM-1465: a scratch probe at `hf` `506ff5b` serialized `HybridOutput { embedding: [NaN, +Inf, -Inf, 1.0], routing_entropy: Some(NaN), .. }` to `{"embedding":[null,null,null,1.0],...,"routing_entropy":null}`. Deserializing that string fails with `invalid type: null, expected f32`. The same holds for `Tensor`. The `Option<f32>` field `routing_entropy` would round-trip `null` to `None` silently. The forward path cannot emit a non-finite embedding or stimuli (rows above), but the struct fields are public and `routing_entropy` is passed through unchecked (`hf:src/reverse.rs:244`), so the type itself does not enforce the invariant.

## 4. Result tracks

**PLANNED, not implemented.** Tracked in [rmems/hybrid-fusion#62](https://github.com/rmems/hybrid-fusion/issues/62). No `ResultTrack` type, field, or enum exists in any of the three crates.

| Wire value | Intended Rust variant (planned) | Meaning |
|------------|---------------------------------|---------|
| `ann` | `Ann` | Transformer / dense result only |
| `snn-native` | `SnnNative` | Spiking network result without a Transformer in the loop |
| `hybrid` | `Hybrid` | Result that crosses the ANN/SNN boundary through hybrid-fusion |

The issue specifies a `#[non_exhaustive]` enum, a required field on telemetry records and forward outputs, and an explicit error (or type error) when two results of different tracks are combined. Until it lands, cross-track comparison is not prevented by any type. Do not compare ANN, native-SNN, and hybrid numbers unless the track is stated in the surrounding report.

## 5. Ownership

| Concern | Owner | Notes |
|---------|-------|-------|
| GGUF / Safetensors parsing, tensor inventory, dim normalization, packed to `f32` decode (Q8_0, Q5_K, Q6_K; IQ3_M only as engram's internal 111-byte block, which cortex does not bind — see §2) | engram-parser | `ct:src/moe/checkpoint.rs:3-13` states cortex does not parse and defers decode to engram |
| Tensor math, reference ANN execution, softmax / norm / activation finite policy, `f32` storage | cortex-tensor | `ct:src/tensor/finite.rs:1-14` |
| Orchestration, stage-graph and port contracts, capability negotiation, ANN/SNN boundary validation, hybrid result types | hybrid-fusion | `hf:src/plan.rs`, `hf:src/capabilities.rs`, `hf:src/hybrid.rs` |
| Result track vocabulary | hybrid-fusion (planned) | #62 |

## Known gaps

1. **No dtype conversion exists** between engram `TensorDType`, cortex `TensorKind`, and hf `Dtype`. Section 2 is a name reading. hf cannot represent GGUF packed types, `I4`/`U4`, or any quantized layout.
2. **cortex binds only F32 gate and F16/Q8_0/Q5_K/Q6_K token rows.** BF16, F64, integers, and all other packed types hit `Other` and fail. IQ3_S has a variant but is an explicit error.
3. **Two cortex dtype enums** (`TensorKind` private to `moe/checkpoint.rs`, `stage::DType` public) share no mapping. `stage::DType` includes `F16`/`BF16` that the reference backend does not execute.
4. **hf `Dtype` is Safetensors vocabulary and has variants engram never produces** (`F4`, `F6*`, `F8*Fnuz`, `F8E8M0`, `C64`).
5. **Finite policy is not uniform at the checkpoint boundary.** engram and cortex bind pass NaN/Inf from file bytes straight through; hf rejects them later at the forward path. A checkpoint with a NaN weight is accepted until the math runs.
6. **Finite policy conflicts at the math boundary.** cortex propagates NaN and defines `+-Inf` outputs; hf rejects any non-finite value at its boundary. No document says which side is authoritative when cortex hands hf a non-finite tensor.
7. **JSON round-trip of non-finite values is lossy and one-way** (section 3): `null` on write, error on read, sign of Inf lost, `Option<f32>` fields silently become `None`. `HybridOutput.routing_entropy` is not validated before serialization.
8. **Reverse-path embedding finiteness not traced.** `hf:src/reverse.rs:149` builds the embedding via `project_spike_activity`; only `hf:src/projector.rs:129` (potentials) was seen checking `is_finite`. Needs confirmation before claiming reverse outputs are always finite.
9. **cortex `Tensor` JSON behavior with NaN not probed.** Section 3 verified hf only; cortex uses a custom `Serialize` (`ct:src/tensor/mod.rs:69-111`).
10. **`PortSpec` axis order is undocumented.** It is positional, but nothing states outermost-first, and engram's innermost-first GGUF dims could be mistaken for a port shape. A stale comment at `ct:src/moe/checkpoint.rs:330` says "`dims` is innermost-first" next to a 1-D `[hidden]` call; that is correct for the dequant call but contradicts the module's outermost-first binding.
11. **Result tracks absent** (section 4). `hf:src/telemetry.rs` is Sentry error capture only (`:1-35`); there is no telemetry record type for a track to attach to yet, which #62 assumes.
12. **Version skew.** cortex binds published `engram-parser` 0.3.0 (`ct:Cargo.toml:61`); the local engram clone at `dbc46cf` also reports 0.3.0, but the match is by version number, not commit. The hf smoke harness pins cortex-tensor by git rev `5cbee81`, not `a0fd18e` (`hf:smoke/cortex-neuromod/Cargo.toml:26`), so it predates PR #124.
