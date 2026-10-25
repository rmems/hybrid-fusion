// SPDX-License-Identifier: MIT OR Apache-2.0

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

impl Tensor {
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        assert!(
            shape.iter().all(|&d| d > 0),
            "Tensor::from_vec: shape dimensions must be > 0, got {shape:?}",
        );
        let expected: usize = shape.iter().product();
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

    pub fn zeros(shape: &[usize]) -> Self {
        assert!(
            shape.iter().all(|&d| d > 0),
            "Tensor::zeros: shape dimensions must be > 0, got {shape:?}",
        );
        let len: usize = shape.iter().product();
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

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
