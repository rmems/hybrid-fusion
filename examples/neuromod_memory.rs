// SPDX-License-Identifier: MIT OR Apache-2.0

//! One 16,384-input / 8-output construction, without stepping.
//! Build first, then measure the binary (not Cargo) with `/usr/bin/time -v`.
//! The printed matrix payload is not measured RSS or total heap allocation.

use hybrid_fusion::{NeuromodSnn, SpikingNetwork};

fn main() -> hybrid_fusion::Result<()> {
    let snn = std::hint::black_box(NeuromodSnn::with_seed(16_384, 8, 42)?);
    println!(
        "inputs={} outputs={} matrix_payload_bytes={}",
        snn.num_channels(),
        snn.num_neurons(),
        NeuromodSnn::pre_inference_matrix_bytes(snn.num_channels(), snn.num_neurons())
    );
    Ok(())
}
