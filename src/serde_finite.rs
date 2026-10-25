// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fail-closed `serde` policy for `f32` fields on JSON-like wire formats
//! (RM-1465).
//!
//! JSON is **not a lossless `f32` container**: `serde_json` has no spelling
//! for `NaN` or ±`Infinity` and silently writes them as `null`, so
//! `serde_json::to_string(&f32::NAN)` produces `"null"`. Left unchecked, that
//! destroys type and meaning on the wire for every record this crate
//! serializes — `embedding: vec![NaN]` becomes `"embedding":[null]` and
//! `routing_entropy: Some(NaN)` becomes `"routing_entropy":null`, which
//! deserializes back as `None`. Downstream logs, Sentry breadcrumbs, and
//! checkpointed fusion traces end up silently wrong.
//!
//! This module implements the crate policy instead of letting that happen:
//! the `Serialize` impls of the `f32`-bearing record types
//! ([`crate::HybridOutput`], [`crate::ExpertRouteOutput`],
//! [`crate::SpikingRouteOutput`], [`crate::SpikeActivity`],
//! [`crate::NeuroModulators`], [`crate::Tensor`]) **reject non-finite values
//! with a `serde` error that names the field and the value** rather than
//! emitting `null`. Failing closed is deliberate; use
//! [`serde_json::to_string`](serde_json::to_string) and the error surfaces as
//! a `serde_json::Error`, which [`crate::HybridError::Json`] wraps via `?`.
//!
//! # Write-side only
//!
//! Only serialization is guarded. Strict JSON cannot express a non-finite
//! number in the first place (`serde_json` rejects `NaN` / `Infinity` literals
//! at parse time), so deserialization needs no equivalent check. The
//! consequence for `Option<f32>` fields is an accepted, documented asymmetry:
//! `null` decodes to `None`, and because a record written under this policy
//! can only contain `null` where the in-memory value *was* `None`, new records
//! are unambiguous. Records written by pre-policy code are not; re-derive the
//! value instead of trusting a legacy `null`.
//!
//! Callers that genuinely need to persist non-finite values must not use these
//! record types: persist the raw buffers with a binary format instead.

/// Wire-facing name of a non-finite `f32`, or `None` when the value is finite.
fn kind(value: f32) -> Option<&'static str> {
    if value.is_nan() {
        Some("NaN")
    } else if value == f32::INFINITY {
        Some("+Inf")
    } else if value == f32::NEG_INFINITY {
        Some("-Inf")
    } else {
        None
    }
}

/// Reason appended to every non-finite rejection, kept identical across fields.
const REASON: &str =
    "refusing to serialize a non-finite f32 (JSON would emit null; see HybridOutput serde policy)";

/// Reject a non-finite `f32` scalar before it reaches a serializer.
///
/// `field` is the dotted record path used in the error message, for example
/// `"HybridOutput.routing_entropy"`.
pub(crate) fn check_f32<E: serde::ser::Error>(value: f32, field: &'static str) -> Result<(), E> {
    match kind(value) {
        Some(kind) => Err(E::custom(format!("{field} is {kind}: {REASON}"))),
        None => Ok(()),
    }
}

/// Reject any non-finite element of an `f32` slice before it reaches a
/// serializer. `field` is the dotted record path, for example
/// `"HybridOutput.embedding"`; the error also carries the offending index.
pub(crate) fn check_f32_slice<E: serde::ser::Error>(
    values: &[f32],
    field: &'static str,
) -> Result<(), E> {
    for (index, &value) in values.iter().enumerate() {
        if let Some(kind) = kind(value) {
            return Err(E::custom(format!("{field}[{index}] is {kind}: {REASON}")));
        }
    }
    Ok(())
}

/// [`check_f32`] for an optional scalar; `None` is always accepted.
pub(crate) fn check_opt_f32<E: serde::ser::Error>(
    value: Option<f32>,
    field: &'static str,
) -> Result<(), E> {
    match value {
        Some(value) => check_f32(value, field),
        None => Ok(()),
    }
}

/// [`check_f32_slice`] for an optional vector; `None` is always accepted.
pub(crate) fn check_opt_f32_slice<E: serde::ser::Error>(
    values: &Option<Vec<f32>>,
    field: &'static str,
) -> Result<(), E> {
    match values {
        Some(values) => check_f32_slice(values, field),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Error type used by `serde_json`, i.e. what callers actually observe.
    type JsonError = serde_json::Error;

    #[test]
    fn kind_names_every_non_finite_spelling() {
        assert_eq!(kind(f32::NAN), Some("NaN"));
        assert_eq!(kind(f32::INFINITY), Some("+Inf"));
        assert_eq!(kind(f32::NEG_INFINITY), Some("-Inf"));
        assert_eq!(kind(0.0), None);
        assert_eq!(kind(-1.5), None);
        assert_eq!(kind(f32::MIN_POSITIVE), None);
    }

    #[test]
    fn check_f32_accepts_finite_and_rejects_each_non_finite() {
        for value in [0.0, -0.0, 1.5, -7.25, f32::MIN, f32::MAX] {
            check_f32::<JsonError>(value, "Record.field")
                .unwrap_or_else(|e| panic!("finite {value} rejected: {e}"));
        }
        for (value, kind) in [
            (f32::NAN, "NaN"),
            (f32::INFINITY, "+Inf"),
            (f32::NEG_INFINITY, "-Inf"),
        ] {
            let err = check_f32::<JsonError>(value, "Record.field").unwrap_err();
            assert_eq!(
                err.to_string(),
                format!("Record.field is {kind}: {REASON}"),
                "unexpected message for {value}"
            );
        }
    }

    #[test]
    fn check_f32_slice_reports_the_first_offending_index() {
        check_f32_slice::<JsonError>(&[1.0, -2.0], "Record.field").unwrap();
        let err = check_f32_slice::<JsonError>(&[1.0, f32::NAN, f32::INFINITY], "Record.field")
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            format!("Record.field[1] is NaN: {REASON}"),
            "first non-finite element wins"
        );
        let err = check_f32_slice::<JsonError>(&[f32::NEG_INFINITY], "Record.field").unwrap_err();
        assert!(err.to_string().starts_with("Record.field[0] is -Inf"));
    }

    #[test]
    fn check_f32_slice_accepts_empty() {
        check_f32_slice::<JsonError>(&[], "Record.field").unwrap();
    }

    #[test]
    fn option_checks_pass_through_none() {
        check_opt_f32::<JsonError>(None, "Record.field").unwrap();
        check_opt_f32_slice::<JsonError>(&None, "Record.field").unwrap();
    }

    #[test]
    fn option_checks_reject_a_non_finite_some() {
        let err = check_opt_f32::<JsonError>(Some(f32::INFINITY), "Record.field").unwrap_err();
        assert_eq!(err.to_string(), format!("Record.field is +Inf: {REASON}"));
        let err = check_opt_f32_slice::<JsonError>(&Some(vec![0.0, f32::NAN]), "Record.field")
            .unwrap_err();
        assert_eq!(err.to_string(), format!("Record.field[1] is NaN: {REASON}"));
    }
}
