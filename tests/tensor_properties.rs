// SPDX-License-Identifier: MIT OR Apache-2.0

use hybrid_fusion::projector::{embed_to_stimuli, embed_to_stimuli_with_width};
use hybrid_fusion::tensor::Tensor;
use proptest::prelude::*;

// ── Tensor shape invariants ──────────────────────────────────────────────────

proptest! {
    /// Every generated shape contains at least one zero dimension, so the
    /// structured error path is exercised on 100% of cases (no no-op samples).
    #[test]
    fn from_vec_rejects_zero_dimensions(
        mut shape in prop::collection::vec(1usize..10, 1..6),
        zero_index in 0..6usize,
    ) {
        let idx = zero_index % shape.len();
        shape[idx] = 0;
        // Product of a shape with a zero dim is 0, so data is empty.
        let result = Tensor::from_vec(vec![], &shape);
        prop_assert!(result.is_err(), "from_vec should reject zero-dim shape");
    }

    #[test]
    fn data_len_equals_shape_product(dims in prop::collection::vec(1usize..8, 1..5)) {
        let len: usize = dims.iter().product();
        let data: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let t = Tensor::from_vec(data, &dims).unwrap();
        // Independent invariants: round-tripped shape equals input, ndim matches.
        prop_assert_eq!(t.shape(), dims.as_slice());
        prop_assert_eq!(t.ndim(), dims.len());
        prop_assert_eq!(t.numel(), len);
        prop_assert_eq!(t.data().len(), len);
    }

    /// Inject a zero so every sample exercises the rejection path.
    #[test]
    fn zeros_rejects_zero_dimensions(
        mut shape in prop::collection::vec(1usize..10, 1..6),
        zero_index in 0..6usize,
    ) {
        let idx = zero_index % shape.len();
        shape[idx] = 0;
        let result = Tensor::zeros(&shape);
        prop_assert!(result.is_err(), "zeros should reject zero-dim shape");
    }
}

// ── Projector invariants ─────────────────────────────────────────────────────

proptest! {
    #[test]
    fn embed_to_stimuli_output_bounded(data in prop::collection::vec(-1e6f32..1e6, 1..128)) {
        let len = data.len();
        // 1-D
        let t1 = Tensor::from_vec(data.clone(), &[len]).unwrap();
        let out1 = embed_to_stimuli(&t1);
        for v in &out1 {
            prop_assert!(v.abs() <= 1.0 + 1e-6, "1-D bound violated: {}", v);
        }

        // 2-D (split into 2 rows if even length)
        if len >= 2 && len % 2 == 0 {
            let rows = 2;
            let cols = len / rows;
            let t2 = Tensor::from_vec(data.clone(), &[rows, cols]).unwrap();
            let out2 = embed_to_stimuli(&t2);
            for v in &out2 {
                prop_assert!(v.abs() <= 1.0 + 1e-6, "2-D bound violated: {}", v);
            }
        }

        // 0-D scalar — requires exactly 1 element
        let t0 = Tensor::from_vec(vec![data[0]], &[]).unwrap();
        let out0 = embed_to_stimuli(&t0);
        for v in &out0 {
            prop_assert!(v.abs() <= 1.0 + 1e-6, "0-D bound violated: {}", v);
        }
    }

    #[test]
    fn embed_to_stimuli_with_width_exact_length(
        data in prop::collection::vec(-1e4f32..1e4, 1..64),
        width in 0usize..128,
    ) {
        let len = data.len();
        let t = Tensor::from_vec(data, &[len]).unwrap();
        let out = embed_to_stimuli_with_width(&t, width);
        prop_assert_eq!(out.len(), width);
        for v in &out {
            prop_assert!(v.abs() <= 1.0 + 1e-6, "bound violated: {}", v);
        }
    }

    /// Generate independent row/col dims so coverage is not stuck at rows=2.
    #[test]
    fn pool_resize_2d_never_panics(
        rows in 2usize..16,
        cols in 1usize..16,
        val in -1e4f32..1e4,
        width in 0usize..128,
    ) {
        let len = rows * cols;
        let data = vec![val; len];
        let t = Tensor::from_vec(data, &[rows, cols]).unwrap();
        let out = embed_to_stimuli_with_width(&t, width);
        prop_assert_eq!(out.len(), width);
        for v in &out {
            prop_assert!(v.abs() <= 1.0 + 1e-6, "bound violated: {}", v);
        }
    }
}
