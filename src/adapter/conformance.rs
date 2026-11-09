// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reusable conformance suite for [`DomainAdapter`] implementations (RM-1811).
//!
//! Every implementation, in this crate or downstream, must pass it:
//!
//! ```
//! use hybrid_fusion::adapter::conformance::assert_domain_adapter_contract;
//! use hybrid_fusion::{PoolResizeTanhAdapter, ProjectionMode};
//!
//! assert_domain_adapter_contract(|| PoolResizeTanhAdapter::new(ProjectionMode::RateSum));
//! ```
//!
//! The suite takes a **factory** rather than an instance: determinism, reset,
//! and failure-atomicity are all statements about how an adapter compares with
//! a freshly built twin. The factory must build adapters that behave
//! identically (same configuration, same seed).
//!
//! # What is checked
//!
//! | Check id | Contract clause |
//! |---|---|
//! | `harness_dims` | the suite's own [`ConformanceDims`] are usable |
//! | `declared_metadata` | `encoding()` / `decoding()` are well formed |
//! | `invalid_input_rejected` | every malformed call returns the documented error from the `*_checked` entry point, without panicking |
//! | `hook_does_not_panic` | the raw hooks never panic on malformed input, and any `Ok` they return still satisfies the output contract |
//! | `valid_input_accepted` | well-formed calls succeed with correctly shaped, finite (and, for stimuli, bounded and in-range) output |
//! | `deterministic` | two fresh adapters fed the same calls agree bitwise |
//! | `reset_restores_initial_state` | after `reset()` the adapter replays a fresh adapter's outputs |
//! | `failure_atomic` | rejected calls (checked entry points, and raw hooks that return `Err`) leave no trace in later outputs |
//!
//! Not checked, because a black-box suite cannot observe it: that the adapter
//! never retains its arguments, and that encode and decode state stay
//! independent where an implementation documents them as independent.

use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};

use super::{
    DomainAdapter, MAX_ADAPTER_WIDTH, STIMULUS_BOUND, validate_embedding, validate_stimuli,
};
use crate::error::{ForwardValueStage, HybridError, Result};
use crate::tensor::Tensor;
use crate::traits::SpikeActivity;

/// Problem sizes the suite drives an adapter with.
///
/// Every field must be in `1..=`[`MAX_ADAPTER_WIDTH`]. Run the suite again
/// with other dims to cover an adapter's width-dependent paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConformanceDims {
    /// SNN input channel count passed to `encode_stimuli`.
    pub snn_width: usize,
    /// SNN output population passed to `decode_activity`.
    pub num_neurons: usize,
    /// Embedding width passed to `decode_activity`.
    pub embed_dim: usize,
}

impl Default for ConformanceDims {
    fn default() -> Self {
        Self {
            snn_width: 8,
            num_neurons: 6,
            embed_dim: 5,
        }
    }
}

/// The first contract clause an adapter broke.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractViolation {
    /// Check id; see the table in the [module docs](self).
    pub check: &'static str,
    /// What happened, including the offending input where useful.
    pub detail: String,
}

impl fmt::Display for ContractViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.check, self.detail)
    }
}

impl std::error::Error for ContractViolation {}

fn violation(check: &'static str, detail: impl Into<String>) -> ContractViolation {
    ContractViolation {
        check,
        detail: detail.into(),
    }
}

/// Panic with the [`ContractViolation`] if `make()` builds an adapter that
/// breaks the [`DomainAdapter`] contract. Use inside `#[test]` functions.
pub fn assert_domain_adapter_contract<A, F>(make: F)
where
    A: DomainAdapter,
    F: Fn() -> A,
{
    if let Err(v) = check_domain_adapter_contract(make) {
        panic!("DomainAdapter contract violation: {v}");
    }
}

/// Run the suite with [`ConformanceDims::default`].
pub fn check_domain_adapter_contract<A, F>(make: F) -> std::result::Result<(), ContractViolation>
where
    A: DomainAdapter,
    F: Fn() -> A,
{
    check_domain_adapter_contract_with(make, ConformanceDims::default())
}

/// Run the suite with explicit [`ConformanceDims`], returning the first
/// violation.
pub fn check_domain_adapter_contract_with<A, F>(
    make: F,
    dims: ConformanceDims,
) -> std::result::Result<(), ContractViolation>
where
    A: DomainAdapter,
    F: Fn() -> A,
{
    for (name, value) in [
        ("snn_width", dims.snn_width),
        ("num_neurons", dims.num_neurons),
        ("embed_dim", dims.embed_dim),
    ] {
        if value == 0 || value > MAX_ADAPTER_WIDTH {
            return Err(violation(
                "harness_dims",
                format!("{name} = {value} is outside 1..={MAX_ADAPTER_WIDTH}"),
            ));
        }
    }

    check_declared_metadata(&make())?;
    let invalid = invalid_calls(dims);
    check_invalid_rejected(&make, &invalid)?;

    let fresh = trace(&mut make(), dims, None)?;
    let again = trace(&mut make(), dims, None)?;
    if let Some(i) = first_difference(&fresh, &again) {
        return Err(violation(
            "deterministic",
            format!("call #{i} differs between two fresh adapters given identical calls"),
        ));
    }

    check_hooks(&make, dims, &invalid, &fresh)?;

    let mut reused = make();
    trace(&mut reused, dims, None)?;
    reused.reset();
    let after_reset = trace(&mut reused, dims, None)?;
    if let Some(i) = first_difference(&fresh, &after_reset) {
        return Err(violation(
            "reset_restores_initial_state",
            format!("call #{i} after reset() differs from a freshly built adapter"),
        ));
    }

    let mut noisy = make();
    let noisy_trace = trace(&mut noisy, dims, Some(&invalid))?;
    if let Some(i) = first_difference(&fresh, &noisy_trace) {
        return Err(violation(
            "failure_atomic",
            format!("call #{i} changed because rejected calls ran before it"),
        ));
    }
    Ok(())
}

fn check_declared_metadata<A: DomainAdapter>(
    adapter: &A,
) -> std::result::Result<(), ContractViolation> {
    let encoding = adapter.encoding();
    if encoding.timesteps == 0 {
        return Err(violation(
            "declared_metadata",
            "encoding().timesteps must be >= 1",
        ));
    }
    if let Some((lo, hi)) = encoding.range
        && !(lo.is_finite() && hi.is_finite() && lo <= hi)
    {
        return Err(violation(
            "declared_metadata",
            format!("encoding().range ({lo}, {hi}) must be finite with lo <= hi"),
        ));
    }
    if adapter.decoding().window == 0 {
        return Err(violation(
            "declared_metadata",
            "decoding().window must be >= 1",
        ));
    }
    Ok(())
}

/// One malformed call and the error its `*_checked` entry point must return.
struct InvalidCall {
    name: &'static str,
    call: Call,
    expect: fn(&HybridError) -> bool,
}

enum Call {
    Encode(Tensor, usize),
    Decode(SpikeActivity, usize, usize),
}

fn apply_checked<A: DomainAdapter>(adapter: &mut A, call: &Call) -> Result<Vec<f32>> {
    match call {
        Call::Encode(hidden, width) => adapter.encode_stimuli_checked(hidden, *width),
        Call::Decode(activity, n, d) => adapter.decode_activity_checked(activity, *n, *d),
    }
}

fn apply_hook<A: DomainAdapter>(adapter: &mut A, call: &Call) -> Result<Vec<f32>> {
    match call {
        Call::Encode(hidden, width) => adapter.encode_stimuli(hidden, *width),
        Call::Decode(activity, n, d) => adapter.decode_activity(activity, *n, *d),
    }
}

fn validate_output(call: &Call, output: &[f32]) -> Result<()> {
    match call {
        Call::Encode(_, width) => validate_stimuli(output, *width),
        Call::Decode(_, _, dim) => validate_embedding(output, *dim),
    }
}

/// Tensor whose `shape` and data are set independently, as serde allows.
fn raw_tensor(data: Vec<f32>, shape: &[usize]) -> Tensor {
    serde_json::from_value(serde_json::json!({ "data": data, "shape": shape }))
        .expect("Tensor deserializes any finite data with any shape")
}

fn invalid_calls(dims: ConformanceDims) -> Vec<InvalidCall> {
    let ConformanceDims {
        snn_width: w,
        num_neurons: n,
        embed_dim: d,
    } = dims;
    let ok_hidden = || Tensor::from_vec(vec![0.1; 6], &[2, 3]);
    let ok_activity = || SpikeActivity::from_fired(&[0], n).expect("n >= 1");

    let mut nan_hidden = ok_hidden();
    nan_hidden.data_mut()[4] = f32::NAN;
    let mut inf_hidden = ok_hidden();
    inf_hidden.data_mut()[0] = f32::NEG_INFINITY;

    let mut nan_potential = ok_activity();
    nan_potential.potentials[0] = f32::NAN;
    let mut out_of_range = ok_activity();
    out_of_range.spike_train = vec![vec![n]];
    let mut repeated = ok_activity();
    repeated.spike_train = vec![vec![0, 0]];
    let mut short_potentials = ok_activity();
    short_potentials.potentials.pop();
    let mut bad_iz = ok_activity();
    bad_iz.iz_potentials = vec![0.0; n + 1];

    vec![
        InvalidCall {
            name: "encode: zero snn_width",
            call: Call::Encode(ok_hidden(), 0),
            expect: |e| matches!(e, HybridError::ZeroSnnChannels),
        },
        InvalidCall {
            name: "encode: snn_width above MAX_ADAPTER_WIDTH",
            call: Call::Encode(ok_hidden(), MAX_ADAPTER_WIDTH + 1),
            expect: |e| matches!(e, HybridError::InvalidConfig(_)),
        },
        InvalidCall {
            name: "encode: rank 0",
            call: Call::Encode(Tensor::from_vec(vec![1.0], &[]), w),
            expect: |e| matches!(e, HybridError::HiddenStateRank { got: 0 }),
        },
        InvalidCall {
            name: "encode: rank 3",
            call: Call::Encode(Tensor::from_vec(vec![0.1; 8], &[2, 2, 2]), w),
            expect: |e| matches!(e, HybridError::HiddenStateRank { got: 3 }),
        },
        InvalidCall {
            name: "encode: zero sequence axis",
            call: Call::Encode(raw_tensor(vec![], &[0, 4]), w),
            expect: |e| {
                matches!(
                    e,
                    HybridError::HiddenStateSeqLen {
                        expected: 1,
                        got: 0
                    }
                )
            },
        },
        InvalidCall {
            name: "encode: zero dim axis",
            call: Call::Encode(raw_tensor(vec![], &[3, 0]), w),
            expect: |e| {
                matches!(
                    e,
                    HybridError::HiddenStateDim {
                        expected: 1,
                        got: 0
                    }
                )
            },
        },
        InvalidCall {
            name: "encode: data shorter than shape",
            call: Call::Encode(raw_tensor(vec![0.1; 5], &[2, 3]), w),
            expect: |e| {
                matches!(
                    e,
                    HybridError::HiddenStateDataLen {
                        expected: 6,
                        got: 5
                    }
                )
            },
        },
        InvalidCall {
            name: "encode: data longer than shape",
            call: Call::Encode(raw_tensor(vec![0.1; 7], &[2, 3]), w),
            expect: |e| {
                matches!(
                    e,
                    HybridError::HiddenStateDataLen {
                        expected: 6,
                        got: 7
                    }
                )
            },
        },
        InvalidCall {
            name: "encode: shape product overflows usize",
            call: Call::Encode(raw_tensor(vec![], &[usize::MAX, 2]), w),
            expect: |e| matches!(e, HybridError::HiddenStateDataLen { .. }),
        },
        InvalidCall {
            name: "encode: NaN hidden state",
            call: Call::Encode(nan_hidden, w),
            expect: |e| {
                matches!(
                    e,
                    HybridError::NonFinite {
                        stage: ForwardValueStage::HiddenState,
                        index: 4
                    }
                )
            },
        },
        InvalidCall {
            name: "encode: -Inf hidden state",
            call: Call::Encode(inf_hidden, w),
            expect: |e| {
                matches!(
                    e,
                    HybridError::NonFinite {
                        stage: ForwardValueStage::HiddenState,
                        index: 0
                    }
                )
            },
        },
        InvalidCall {
            name: "decode: zero embed_dim",
            call: Call::Decode(ok_activity(), n, 0),
            expect: |e| matches!(e, HybridError::InvalidConfig(_)),
        },
        InvalidCall {
            name: "decode: embed_dim above MAX_ADAPTER_WIDTH",
            call: Call::Decode(ok_activity(), n, MAX_ADAPTER_WIDTH + 1),
            expect: |e| matches!(e, HybridError::InvalidConfig(_)),
        },
        InvalidCall {
            name: "decode: zero num_neurons",
            call: Call::Decode(ok_activity(), 0, d),
            expect: |e| matches!(e, HybridError::InvalidConfig(_)),
        },
        InvalidCall {
            name: "decode: num_neurons above MAX_ADAPTER_WIDTH",
            call: Call::Decode(ok_activity(), MAX_ADAPTER_WIDTH + 1, d),
            expect: |e| matches!(e, HybridError::InvalidConfig(_)),
        },
        InvalidCall {
            name: "decode: potentials shorter than num_neurons",
            call: Call::Decode(short_potentials, n, d),
            expect: |e| matches!(e, HybridError::InputLengthMismatch { .. }),
        },
        InvalidCall {
            name: "decode: non-finite potential",
            call: Call::Decode(nan_potential, n, d),
            expect: |e| matches!(e, HybridError::InvalidConfig(_)),
        },
        InvalidCall {
            name: "decode: spike index out of range",
            call: Call::Decode(out_of_range, n, d),
            expect: |e| matches!(e, HybridError::InvalidConfig(_)),
        },
        InvalidCall {
            name: "decode: repeated spike index in one step",
            call: Call::Decode(repeated, n, d),
            expect: |e| matches!(e, HybridError::InvalidConfig(_)),
        },
        InvalidCall {
            name: "decode: iz_potentials of the wrong length",
            call: Call::Decode(bad_iz, n, d),
            expect: |e| matches!(e, HybridError::InputLengthMismatch { .. }),
        },
    ]
}

fn check_invalid_rejected<A, F>(
    make: &F,
    invalid: &[InvalidCall],
) -> std::result::Result<(), ContractViolation>
where
    A: DomainAdapter,
    F: Fn() -> A,
{
    for case in invalid {
        let mut adapter = make();
        let outcome = catch_unwind(AssertUnwindSafe(|| apply_checked(&mut adapter, &case.call)));
        match outcome {
            Err(_) => {
                return Err(violation(
                    "invalid_input_rejected",
                    format!("{}: the checked entry point panicked", case.name),
                ));
            }
            Ok(Ok(_)) => {
                return Err(violation(
                    "invalid_input_rejected",
                    format!("{}: accepted, expected an error", case.name),
                ));
            }
            Ok(Err(err)) if !(case.expect)(&err) => {
                return Err(violation(
                    "invalid_input_rejected",
                    format!("{}: wrong error variant: {err:?}", case.name),
                ));
            }
            Ok(Err(_)) => {}
        }
    }
    Ok(())
}

/// Raw hooks, called directly with malformed input: they must not panic, any
/// `Ok` must still satisfy the output contract, and an `Err` must leave the
/// adapter indistinguishable from a fresh one (compared against `fresh`).
fn check_hooks<A, F>(
    make: &F,
    dims: ConformanceDims,
    invalid: &[InvalidCall],
    fresh: &Trace,
) -> std::result::Result<(), ContractViolation>
where
    A: DomainAdapter,
    F: Fn() -> A,
{
    for case in invalid {
        let mut adapter = make();
        let outcome = catch_unwind(AssertUnwindSafe(|| apply_hook(&mut adapter, &case.call)));
        match outcome {
            Err(_) => {
                return Err(violation(
                    "hook_does_not_panic",
                    format!("{}: the implementation hook panicked", case.name),
                ));
            }
            Ok(Ok(output)) => {
                if let Err(err) = validate_output(&case.call, &output) {
                    return Err(violation(
                        "hook_does_not_panic",
                        format!(
                            "{}: hook returned Ok with an out-of-contract output: {err}",
                            case.name
                        ),
                    ));
                }
            }
            Ok(Err(_)) => {
                let after = trace(&mut adapter, dims, None)?;
                if let Some(i) = first_difference(fresh, &after) {
                    return Err(violation(
                        "failure_atomic",
                        format!(
                            "{}: hook returned Err but call #{i} afterwards differs from a fresh adapter",
                            case.name
                        ),
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Bit patterns of every output, one entry per call.
type Trace = Vec<Vec<u32>>;

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn first_difference(a: &Trace, b: &Trace) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter().zip(b).position(|(x, y)| x != y)
}

fn hidden_samples(width: usize) -> Vec<Tensor> {
    let wave = |len: usize, scale: f32| -> Vec<f32> {
        (0..len).map(|i| (i as f32 * 0.37).sin() * scale).collect()
    };
    vec![
        Tensor::from_vec(wave(7, 3.0), &[7]),
        Tensor::from_vec(wave(15, 1.5), &[3, 5]),
        Tensor::from_vec(vec![0.25], &[1]),
        Tensor::from_vec(wave(2 * (width + 3), 0.8), &[2, width + 3]),
        Tensor::from_vec(vec![1.0e20, -2.0e20, 5.0e19], &[3]),
    ]
}

fn activity_samples(n: usize) -> Vec<SpikeActivity> {
    let ramp: Vec<f32> = (0..n)
        .map(|i| (i as f32 + 1.0) / (n as f32 + 1.0))
        .collect();
    let fired: Vec<usize> = if n > 1 { vec![0, n - 1] } else { vec![0] };
    let mut sparse = SpikeActivity::from_fired(&fired, n).expect("in range, distinct");
    sparse.potentials.clone_from(&ramp);
    let train: Vec<Vec<usize>> = (0..5)
        .map(|t| (0..n).filter(|i| (i + t) % 3 == 0).collect())
        .collect();
    vec![
        SpikeActivity {
            spike_train: Vec::new(),
            potentials: vec![0.0; n],
            iz_potentials: Vec::new(),
        },
        sparse,
        SpikeActivity {
            spike_train: train,
            potentials: ramp.clone(),
            iz_potentials: ramp,
        },
    ]
}

/// Drive one adapter through a fixed valid call sequence, recording every
/// output and enforcing the output contract on each. With `noise`, every
/// malformed call is issued (and must be rejected) before each valid call.
fn trace<A: DomainAdapter>(
    adapter: &mut A,
    dims: ConformanceDims,
    noise: Option<&[InvalidCall]>,
) -> std::result::Result<Trace, ContractViolation> {
    let range = adapter.encoding().range;
    let hidden = hidden_samples(dims.snn_width);
    let activities = activity_samples(dims.num_neurons);
    let mut out = Trace::new();

    let reject_noise = |adapter: &mut A| -> std::result::Result<(), ContractViolation> {
        for case in noise.unwrap_or_default() {
            let outcome = catch_unwind(AssertUnwindSafe(|| apply_checked(adapter, &case.call)));
            if !matches!(outcome, Ok(Err(_))) {
                return Err(violation(
                    "failure_atomic",
                    format!("{}: was not rejected while probing atomicity", case.name),
                ));
            }
        }
        Ok(())
    };

    // `hidden[0]` is encoded twice so any call-history dependence shows up.
    let encode_order: Vec<&Tensor> = hidden.iter().chain(std::iter::once(&hidden[0])).collect();
    for (i, h) in encode_order.into_iter().enumerate() {
        reject_noise(adapter)?;
        let stimuli = adapter
            .encode_stimuli_checked(h, dims.snn_width)
            .map_err(|e| violation("valid_input_accepted", format!("encode sample {i}: {e}")))?;
        if let Some((lo, hi)) = range
            && let Some(v) = stimuli.iter().find(|v| **v < lo || **v > hi)
        {
            return Err(violation(
                "declared_metadata",
                format!("encode sample {i}: stimulus {v} is outside declared range ({lo}, {hi})"),
            ));
        }
        debug_assert!(stimuli.iter().all(|v| v.abs() <= STIMULUS_BOUND));
        out.push(bits(&stimuli));
    }
    for (i, activity) in activities.iter().enumerate() {
        reject_noise(adapter)?;
        let embedding = adapter
            .decode_activity_checked(activity, dims.num_neurons, dims.embed_dim)
            .map_err(|e| violation("valid_input_accepted", format!("decode sample {i}: {e}")))?;
        out.push(bits(&embedding));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::PoolResizeTanhAdapter;
    use crate::error::HybridError;
    use crate::types::ProjectionMode;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Reference adapter with one knob per contract clause to break.
    #[derive(Clone, Copy, Default)]
    struct Faulty {
        inner: PoolResizeTanhAdapter,
        panic_on_bad_rank: bool,
        out_of_bound: bool,
        short_output: bool,
        nan_embedding: bool,
        non_deterministic: bool,
        reset_is_noop: bool,
        advances_on_failure: bool,
        state: u32,
        bad_range: bool,
    }

    static GLOBAL_TICKS: AtomicU32 = AtomicU32::new(0);

    impl DomainAdapter for Faulty {
        fn encode_stimuli(&mut self, hidden: &Tensor, snn_width: usize) -> Result<Vec<f32>> {
            if self.panic_on_bad_rank && hidden.ndim() > 2 {
                panic!("rank {} not handled", hidden.ndim());
            }
            if self.advances_on_failure {
                self.state += 1;
            }
            let mut out = self.inner.encode_stimuli(hidden, snn_width)?;
            if self.out_of_bound {
                out[0] = 2.0;
            }
            if self.short_output {
                out.pop();
            }
            if self.non_deterministic {
                out[0] = (GLOBAL_TICKS.fetch_add(1, Ordering::Relaxed) % 7) as f32 / 10.0;
            }
            if self.reset_is_noop || self.advances_on_failure {
                // Output depends on call history through `state`.
                self.state += 1;
                out[0] = (self.state % 5) as f32 / 10.0;
            }
            Ok(out)
        }

        fn decode_activity(
            &mut self,
            activity: &SpikeActivity,
            num_neurons: usize,
            embed_dim: usize,
        ) -> Result<Vec<f32>> {
            let mut out = self
                .inner
                .decode_activity(activity, num_neurons, embed_dim)?;
            if self.nan_embedding {
                out[0] = f32::NAN;
            }
            Ok(out)
        }

        fn reset(&mut self) {
            if !self.reset_is_noop {
                self.state = 0;
            }
        }

        fn encoding(&self) -> crate::SpikeEncoding {
            if self.bad_range {
                crate::SpikeEncoding::new(crate::SpikeEncodingKind::Rate, 1, Some((0.0, 0.0001)))
            } else {
                self.inner.encoding()
            }
        }
    }

    fn violation_of(faulty: Faulty) -> ContractViolation {
        check_domain_adapter_contract(|| faulty).expect_err("adapter must be rejected")
    }

    #[test]
    fn the_faultless_baseline_passes() {
        check_domain_adapter_contract(Faulty::default).expect("baseline conforms");
    }

    #[test]
    fn panicking_hook_is_reported() {
        let v = violation_of(Faulty {
            panic_on_bad_rank: true,
            ..Faulty::default()
        });
        assert_eq!(v.check, "hook_does_not_panic", "{v}");
    }

    #[test]
    fn out_of_bound_stimuli_are_reported() {
        let v = violation_of(Faulty {
            out_of_bound: true,
            ..Faulty::default()
        });
        assert_eq!(v.check, "valid_input_accepted", "{v}");
    }

    #[test]
    fn wrong_length_output_is_reported() {
        let v = violation_of(Faulty {
            short_output: true,
            ..Faulty::default()
        });
        assert_eq!(v.check, "valid_input_accepted", "{v}");
    }

    #[test]
    fn nan_embedding_is_reported() {
        let v = violation_of(Faulty {
            nan_embedding: true,
            ..Faulty::default()
        });
        assert_eq!(v.check, "valid_input_accepted", "{v}");
    }

    #[test]
    fn checked_entry_points_reject_out_of_contract_output() {
        let hidden = Tensor::from_vec(vec![0.1; 4], &[4]);
        let mut a = Faulty {
            out_of_bound: true,
            ..Faulty::default()
        };
        assert!(matches!(
            a.encode_stimuli_checked(&hidden, 4),
            Err(HybridError::InvalidConfig(_))
        ));
        let mut a = Faulty {
            short_output: true,
            ..Faulty::default()
        };
        assert!(matches!(
            a.encode_stimuli_checked(&hidden, 4),
            Err(HybridError::InputLengthMismatch {
                expected: 4,
                got: 3
            })
        ));
        let mut a = Faulty {
            nan_embedding: true,
            ..Faulty::default()
        };
        let act = SpikeActivity::from_fired(&[0], 2).unwrap();
        assert!(matches!(
            a.decode_activity_checked(&act, 2, 3),
            Err(HybridError::NonFinite {
                stage: ForwardValueStage::Embedding,
                index: 0
            })
        ));
    }

    #[test]
    fn non_determinism_is_reported() {
        let v = violation_of(Faulty {
            non_deterministic: true,
            ..Faulty::default()
        });
        assert_eq!(v.check, "deterministic", "{v}");
    }

    #[test]
    fn state_surviving_reset_is_reported() {
        let v = violation_of(Faulty {
            reset_is_noop: true,
            ..Faulty::default()
        });
        assert_eq!(v.check, "reset_restores_initial_state", "{v}");
    }

    #[test]
    fn state_advanced_by_a_failed_call_is_reported() {
        let v = violation_of(Faulty {
            advances_on_failure: true,
            ..Faulty::default()
        });
        assert_eq!(v.check, "failure_atomic", "{v}");
    }

    #[test]
    fn declared_range_violations_are_reported() {
        let v = violation_of(Faulty {
            bad_range: true,
            ..Faulty::default()
        });
        assert_eq!(v.check, "declared_metadata", "{v}");
    }

    #[test]
    fn unusable_harness_dims_are_reported() {
        for dims in [
            ConformanceDims {
                snn_width: 0,
                ..ConformanceDims::default()
            },
            ConformanceDims {
                num_neurons: MAX_ADAPTER_WIDTH + 1,
                ..ConformanceDims::default()
            },
        ] {
            let v = check_domain_adapter_contract_with(Faulty::default, dims).unwrap_err();
            assert_eq!(v.check, "harness_dims");
        }
    }

    #[test]
    fn reference_adapter_conforms_across_dims() {
        for (snn_width, num_neurons, embed_dim) in [(1, 1, 1), (3, 2, 9), (64, 17, 3)] {
            let dims = ConformanceDims {
                snn_width,
                num_neurons,
                embed_dim,
            };
            check_domain_adapter_contract_with(
                || PoolResizeTanhAdapter::new(ProjectionMode::TemporalHistogram),
                dims,
            )
            .unwrap_or_else(|v| panic!("{dims:?}: {v}"));
        }
    }

    #[test]
    #[should_panic(expected = "DomainAdapter contract violation: [deterministic]")]
    fn assert_helper_panics_with_the_violation() {
        assert_domain_adapter_contract(|| Faulty {
            non_deterministic: true,
            ..Faulty::default()
        });
    }
}
