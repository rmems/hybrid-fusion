// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::HybridError;
use serde::{Deserialize, Serialize};

/// Dense `f32` buffer plus its shape (`Tensor::from_vec` enforces
/// `data.len() == shape.product()` and non-zero extents).
///
/// # JSON / serde policy
///
/// `data` fails closed on non-finite values: serializing a tensor that holds
/// any `NaN` / ±`Inf` returns an error naming `Tensor.data[index]` instead of
/// letting `serde_json` write `null` (crate `serde_finite` policy, RM-1465).
/// Forward-path hidden states are already rejected when non-finite
/// ([`crate::HybridNetwork::forward`](crate::HybridNetwork::forward)), so a
/// hand-built tensor with non-finite data is a contract violation, not a
/// payload to persist.
#[derive(Debug, Clone, Deserialize)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

/// Fails closed on non-finite `f32` values instead of letting `serde_json`
/// write `null` (see [`Tensor`] docs).
impl Serialize for Tensor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        crate::serde_finite::check_f32_slice(&self.data, "Tensor.data")?;

        // Field order mirrors the struct declaration, as the derived impl did.
        let mut state = serializer.serialize_struct("Tensor", 2)?;
        state.serialize_field("data", &self.data)?;
        state.serialize_field("shape", &self.shape)?;
        state.end()
    }
}

/// Product of all shape dimensions, `None` on `usize` overflow. The empty
/// shape yields `1` (rank-0 scalar).
fn shape_numel(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
}

impl Tensor {
    /// Fallible [`Self::from_vec`]: returns [`HybridError::InvalidConfig`]
    /// instead of panicking when a shape dimension is `0`, the shape product
    /// overflows `usize`, or `data.len()` does not equal the shape product.
    /// Backend code on the [`crate::Transformer::hidden_states`] path should
    /// use this (or [`Self::try_zeros`]) so a construction failure surfaces
    /// as `Err`.
    ///
    /// Rank-0 tensors (`shape == &[]`, one element) are valid, matching
    /// [`Self::from_vec`].
    pub fn try_from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self, HybridError> {
        if shape.contains(&0) {
            return Err(HybridError::InvalidConfig(format!(
                "Tensor::try_from_vec: shape dimensions must be > 0, got {shape:?}"
            )));
        }
        let Some(expected) = shape_numel(shape) else {
            return Err(HybridError::InvalidConfig(format!(
                "Tensor::try_from_vec: shape product overflows usize (shape={shape:?})"
            )));
        };
        if data.len() != expected {
            return Err(HybridError::InvalidConfig(format!(
                "Tensor::try_from_vec: data.len()={} does not match shape product={expected} (shape={shape:?})",
                data.len(),
            )));
        }
        Ok(Self {
            data,
            shape: shape.to_vec(),
        })
    }

    /// Fallible [`Self::zeros`]: returns [`HybridError::InvalidConfig`] when
    /// a shape dimension is `0`, the shape product overflows `usize`, or the
    /// buffer cannot be allocated, instead of panicking.
    pub fn try_zeros(shape: &[usize]) -> Result<Self, HybridError> {
        if shape.contains(&0) {
            return Err(HybridError::InvalidConfig(format!(
                "Tensor::try_zeros: shape dimensions must be > 0, got {shape:?}"
            )));
        }
        let Some(len) = shape_numel(shape) else {
            return Err(HybridError::InvalidConfig(format!(
                "Tensor::try_zeros: shape product overflows usize (shape={shape:?})"
            )));
        };
        let mut data = Vec::new();
        data.try_reserve_exact(len).map_err(|_| {
            HybridError::InvalidConfig(format!(
                "Tensor::try_zeros: cannot allocate {len} elements (shape={shape:?})"
            ))
        })?;
        data.resize(len, 0.0);
        Ok(Self {
            data,
            shape: shape.to_vec(),
        })
    }

    /// Panicking convenience wrapper over [`Self::try_from_vec`] for tests and
    /// callers that have already proven the shape is valid. Backend code
    /// should prefer the `try_*` constructors so failures return `Err`.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        assert!(
            shape.iter().all(|&d| d > 0),
            "Tensor::from_vec: shape dimensions must be > 0, got {shape:?}",
        );
        let expected = shape_numel(shape).expect("Tensor::from_vec: shape product overflows usize");
        assert_eq!(
            data.len(),
            expected,
            "Tensor::from_vec: data.len()={} does not match shape product={expected} (shape={shape:?})",
            data.len(),
        );
        Self {
            data,
            shape: shape.to_vec(),
        }
    }

    /// Panicking convenience wrapper over [`Self::try_zeros`] for tests and
    /// callers that have already proven the shape is valid. Backend code
    /// should prefer the `try_*` constructors so failures return `Err`.
    pub fn zeros(shape: &[usize]) -> Self {
        assert!(
            shape.iter().all(|&d| d > 0),
            "Tensor::zeros: shape dimensions must be > 0, got {shape:?}",
        );
        let len = shape_numel(shape).expect("Tensor::zeros: shape product overflows usize");
        Self {
            data: vec![0.0; len],
            shape: shape.to_vec(),
        }
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Number of scalar elements (product of `shape`).
    ///
    /// Saturates at `usize::MAX` when the product overflows `usize` —
    /// reachable only for tensors deserialized from untrusted input, since
    /// the constructors reject overflowing and zero-extent shapes. A shape
    /// containing a `0` axis yields `0`, matching the mathematical product.
    pub fn numel(&self) -> usize {
        if self.shape.contains(&0) {
            return 0;
        }
        shape_numel(&self.shape).unwrap_or(usize::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::HybridError;

    #[test]
    fn try_from_vec_rejects_zero_extent_shapes() {
        for shape in [&[0][..], &[1, 0][..], &[0, 4][..], &[2, 0, 3][..]] {
            let len: usize = shape.iter().product();
            match Tensor::try_from_vec(vec![0.0; len], shape).unwrap_err() {
                HybridError::InvalidConfig(msg) => {
                    assert!(msg.contains("must be > 0"), "unexpected message: {msg}");
                }
                other => panic!("unexpected {other:?}"),
            }
            assert!(
                matches!(Tensor::try_zeros(shape), Err(HybridError::InvalidConfig(_))),
                "try_zeros should reject zero-extent shape {shape:?}"
            );
        }
    }

    #[test]
    fn try_from_vec_rejects_data_len_mismatch() {
        match Tensor::try_from_vec(vec![0.0; 3], &[2, 2]).unwrap_err() {
            HybridError::InvalidConfig(msg) => {
                assert!(msg.contains("does not match shape product"), "{msg}");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn try_constructors_reject_shape_product_overflow() {
        // usize::MAX * usize::MAX overflows; checked_mul returns None on every
        // profile. The shape is chosen so a *wrapped* product would be 1 —
        // without checked_mul a release build would accept this one-element
        // buffer and return a tensor whose shape lies about its data.
        let shape = [usize::MAX, usize::MAX];
        for result in [
            Tensor::try_from_vec(vec![0.0], &shape),
            Tensor::try_zeros(&shape),
        ] {
            assert!(
                matches!(result, Err(HybridError::InvalidConfig(_))),
                "overflowing shape must be InvalidConfig"
            );
        }
    }

    #[test]
    fn try_zeros_rejects_unallocatable_buffer() {
        // The element count fits usize but the f32 buffer exceeds Vec's
        // byte-capacity limit, so try_reserve_exact must fail closed.
        let shape = [usize::MAX / 2];
        assert!(matches!(
            Tensor::try_zeros(&shape),
            Err(HybridError::InvalidConfig(_))
        ));
    }

    #[test]
    fn try_constructors_accept_valid_and_rank0_shapes() {
        let t = Tensor::try_from_vec(vec![0.5, -1.25, 3.0], &[3]).unwrap();
        assert_eq!(t.shape(), &[3]);
        Tensor::try_from_vec(vec![0.0; 6], &[2, 3]).unwrap();
        Tensor::try_zeros(&[2, 3]).unwrap();

        // Rank-0 scalar: empty shape, exactly one element.
        let scalar = Tensor::try_from_vec(vec![3.5], &[]).unwrap();
        assert_eq!(scalar.ndim(), 0);
        assert_eq!(scalar.numel(), 1);
        assert_eq!(scalar.data(), &[3.5]);
        let z = Tensor::try_zeros(&[]).unwrap();
        assert_eq!(z.data(), &[0.0]);
    }

    #[test]
    fn numel_saturates_on_overflowing_deserialized_shape() {
        // Deserialize bypasses the constructors' validation, so a crafted
        // record can carry a shape whose product overflows usize; numel
        // must saturate rather than silently wrap. 65536^4 = 2^64.
        let t: Tensor = serde_json::from_str(r#"{"data":[],"shape":[65536,65536,65536,65536]}"#)
            .expect("fields deserialize without cross-validation");
        assert_eq!(t.numel(), usize::MAX);

        // A zero axis yields 0 even when an earlier prefix overflows.
        let z: Tensor = serde_json::from_str(r#"{"data":[],"shape":[65536,65536,65536,65536,0]}"#)
            .expect("fields deserialize without cross-validation");
        assert_eq!(z.numel(), 0);
    }

    #[test]
    fn tensor_finite_round_trips_through_json() {
        let t = Tensor::from_vec(vec![0.5, -1.25, 3.0], &[3]);
        let json = serde_json::to_string(&t).expect("finite data serializes");
        assert_eq!(json, r#"{"data":[0.5,-1.25,3.0],"shape":[3]}"#);
        let restored: Tensor = serde_json::from_str(&json).expect("round trip");
        assert_eq!(restored.data(), t.data());
        assert_eq!(restored.shape(), t.shape());
    }

    #[test]
    fn tensor_serialization_rejects_non_finite_data() {
        for (value, kind) in [(f32::NAN, "NaN"), (f32::INFINITY, "+Inf")] {
            let mut t = Tensor::from_vec(vec![1.0, 2.0], &[2]);
            t.data_mut()[1] = value;
            let err = serde_json::to_string(&t)
                .expect_err("non-finite data must fail closed, not write null");
            let text = err.to_string();
            assert!(
                text.starts_with(&format!("Tensor.data[1] is {kind}")),
                "unexpected error: {text}"
            );
            assert!(
                text.contains("JSON would emit null"),
                "policy reason in {text}"
            );
        }
    }

    #[test]
    fn tensor_deserialization_rejects_null_elements() {
        // Legacy record shape produced by the pre-policy writer.
        let err = serde_json::from_str::<Tensor>(r#"{"data":[null],"shape":[1]}"#)
            .expect_err("a null element is not an f32");
        assert!(
            err.to_string().contains("invalid type: null, expected f32"),
            "unexpected error: {err}"
        );
    }
}
