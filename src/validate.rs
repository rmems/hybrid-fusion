// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pure preflight checks for the ANN→SNN forward path.
//!
//! These validators inspect transformer hidden-state layout and host SNN width
//! without stepping the SNN or incrementing `global_step`. Contract failures
//! are returned as structured [`HybridError`] variants and must not be reported
//! as Sentry runtime failures.

use crate::error::{ForwardValueStage, HybridError, Result};
use crate::tensor::Tensor;

/// Reject a zero-width SNN before projection or [`crate::SpikingNetwork::step`].
pub(crate) fn validate_snn_width(snn_width: usize) -> Result<()> {
    if snn_width == 0 {
        return Err(HybridError::ZeroSnnChannels);
    }
    Ok(())
}

/// Validate transformer hidden-state rank, axes, storage length, and finiteness.
///
/// `seq_len` is `token_ids.len()` (already known non-empty by the caller).
/// `dim` is [`crate::Transformer::dim`].
pub(crate) fn validate_hidden_state(hidden: &Tensor, seq_len: usize, dim: usize) -> Result<()> {
    match hidden.ndim() {
        1 => validate_rank1(hidden, dim),
        2 => validate_rank2(hidden, seq_len, dim),
        got => Err(HybridError::HiddenStateRank { got }),
    }
}

/// Reject the first NaN or ±Inf in `values`.
pub(crate) fn validate_finite(values: &[f32], stage: ForwardValueStage) -> Result<()> {
    if let Some(index) = values.iter().position(|v| !v.is_finite()) {
        return Err(HybridError::NonFinite { stage, index });
    }
    Ok(())
}

fn validate_rank1(hidden: &Tensor, dim: usize) -> Result<()> {
    let got = hidden.shape()[0];
    if got != dim {
        return Err(HybridError::HiddenStateDim { expected: dim, got });
    }
    validate_storage_and_finiteness(hidden, dim)
}

fn validate_rank2(hidden: &Tensor, seq_len: usize, dim: usize) -> Result<()> {
    let shape = hidden.shape();
    let got_seq = shape[0];
    let got_dim = shape[1];
    if got_seq != seq_len {
        return Err(HybridError::HiddenStateSeqLen {
            expected: seq_len,
            got: got_seq,
        });
    }
    if got_dim != dim {
        return Err(HybridError::HiddenStateDim {
            expected: dim,
            got: got_dim,
        });
    }
    let expected = seq_len
        .checked_mul(dim)
        .ok_or(HybridError::HiddenStateDataLen {
            expected: usize::MAX,
            got: hidden.data().len(),
        })?;
    validate_storage_and_finiteness(hidden, expected)
}

fn validate_storage_and_finiteness(hidden: &Tensor, expected_len: usize) -> Result<()> {
    let got = hidden.data().len();
    if got != expected_len {
        return Err(HybridError::HiddenStateDataLen {
            expected: expected_len,
            got,
        });
    }
    validate_finite(hidden.data(), ForwardValueStage::HiddenState)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const SEQ: usize = 3;
    const DIM: usize = 4;

    fn rank2_finite() -> Tensor {
        let data: Vec<f32> = (0..SEQ * DIM).map(|i| i as f32 * 0.01).collect();
        Tensor::from_vec(data, &[SEQ, DIM])
    }

    fn deserialize_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        serde_json::from_value(json!({ "data": data, "shape": shape })).expect("Tensor deserialize")
    }

    #[test]
    fn rank2_finite_ok() {
        validate_hidden_state(&rank2_finite(), SEQ, DIM).unwrap();
    }

    #[test]
    fn rank1_finite_ok() {
        let t = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[DIM]);
        validate_hidden_state(&t, SEQ, DIM).unwrap();
    }

    #[test]
    fn rejects_rank0() {
        let t = Tensor::from_vec(vec![1.0], &[]);
        match validate_hidden_state(&t, SEQ, DIM).unwrap_err() {
            HybridError::HiddenStateRank { got: 0 } => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn rejects_rank3() {
        let t = Tensor::from_vec(vec![0.0; SEQ * DIM * 2], &[SEQ, DIM, 2]);
        match validate_hidden_state(&t, SEQ, DIM).unwrap_err() {
            HybridError::HiddenStateRank { got: 3 } => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn rejects_rank4() {
        let t = Tensor::from_vec(vec![1.0; 8], &[2, 2, 2, 1]);
        match validate_hidden_state(&t, SEQ, DIM).unwrap_err() {
            HybridError::HiddenStateRank { got: 4 } => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn rejects_wrong_sequence_length() {
        let t = Tensor::from_vec(vec![0.0; (SEQ + 1) * DIM], &[SEQ + 1, DIM]);
        match validate_hidden_state(&t, SEQ, DIM).unwrap_err() {
            HybridError::HiddenStateSeqLen { expected: SEQ, got } if got == SEQ + 1 => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn rejects_wrong_hidden_dim_rank2() {
        let t = Tensor::from_vec(vec![0.0; SEQ * (DIM + 1)], &[SEQ, DIM + 1]);
        match validate_hidden_state(&t, SEQ, DIM).unwrap_err() {
            HybridError::HiddenStateDim { expected: DIM, got } if got == DIM + 1 => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn rejects_wrong_hidden_dim_rank1() {
        let t = Tensor::from_vec(vec![0.0; DIM + 2], &[DIM + 2]);
        match validate_hidden_state(&t, SEQ, DIM).unwrap_err() {
            HybridError::HiddenStateDim { expected: DIM, got } if got == DIM + 2 => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn rejects_empty_backing_data() {
        let t = deserialize_tensor(vec![], vec![SEQ, DIM]);
        match validate_hidden_state(&t, SEQ, DIM).unwrap_err() {
            HybridError::HiddenStateDataLen { expected, got: 0 } if expected == SEQ * DIM => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn rejects_inconsistent_storage() {
        let t = deserialize_tensor(vec![0.0; 5], vec![SEQ, DIM]);
        match validate_hidden_state(&t, SEQ, DIM).unwrap_err() {
            HybridError::HiddenStateDataLen { expected, got: 5 } if expected == SEQ * DIM => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn rejects_nan_in_hidden() {
        let mut t = rank2_finite();
        t.data_mut()[2] = f32::NAN;
        match validate_hidden_state(&t, SEQ, DIM).unwrap_err() {
            HybridError::NonFinite {
                stage: ForwardValueStage::HiddenState,
                index: 2,
            } => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn rejects_pos_inf_in_hidden() {
        let mut t = rank2_finite();
        t.data_mut()[0] = f32::INFINITY;
        match validate_hidden_state(&t, SEQ, DIM).unwrap_err() {
            HybridError::NonFinite {
                stage: ForwardValueStage::HiddenState,
                index: 0,
            } => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn rejects_neg_inf_in_hidden() {
        let mut t = rank2_finite();
        t.data_mut()[5] = f32::NEG_INFINITY;
        match validate_hidden_state(&t, SEQ, DIM).unwrap_err() {
            HybridError::NonFinite {
                stage: ForwardValueStage::HiddenState,
                index: 5,
            } => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn rejects_zero_snn_width() {
        match validate_snn_width(0).unwrap_err() {
            HybridError::ZeroSnnChannels => {}
            other => panic!("unexpected {other:?}"),
        }
        validate_snn_width(1).unwrap();
    }

    #[test]
    fn validate_finite_names_stage() {
        match validate_finite(&[1.0, f32::NAN], ForwardValueStage::Embedding).unwrap_err() {
            HybridError::NonFinite {
                stage: ForwardValueStage::Embedding,
                index: 1,
            } => {}
            other => panic!("unexpected {other:?}"),
        }
        match validate_finite(&[f32::INFINITY], ForwardValueStage::Stimuli).unwrap_err() {
            HybridError::NonFinite {
                stage: ForwardValueStage::Stimuli,
                index: 0,
            } => {}
            other => panic!("unexpected {other:?}"),
        }
    }
}
