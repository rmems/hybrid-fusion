// SPDX-License-Identifier: MIT OR Apache-2.0
//
// End-to-end hybrid forward pass using the reference backends.
// Run with:  cargo run --features backends --example simple_hybrid

use hybrid_fusion::backends::simple_snn::SimpleSnn;
use hybrid_fusion::backends::simple_transformer::SimpleTransformer;
use hybrid_fusion::traits::NeuroModulators;
use hybrid_fusion::{HybridConfig, HybridNetwork};

fn main() {
    let config = HybridConfig::tiny();

    let transformer = SimpleTransformer::new(config.transformer.clone());
    let snn = SimpleSnn::new(config.snn_input_channels);

    let mut network =
        HybridNetwork::try_new(transformer, snn, config).expect("config matches backends");

    let token_ids: Vec<u32> = vec![42, 17, 8, 99];
    let modulators = NeuroModulators {
        dopamine: 0.6,
        cortisol: 0.1,
        acetylcholine: 0.7,
        tempo: 1.2,
        aux_dopamine: 0.0,
    };

    println!("Token IDs:     {token_ids:?}");
    println!("Modulators:    {modulators:?}");

    let output = network
        .forward(&token_ids, Some(modulators))
        .expect("forward pass failed");

    println!("Embedding dim: {}", output.embedding.len());
    println!("Stimuli dim:   {}", output.stimuli.len());
    println!("Fired neurons: {:?}", output.fired_neurons);
    println!("Global step:   {}", output.global_step);

    // Second pass -- global_step should increment and membrane state carries over.
    let output2 = network
        .forward(&[1, 2, 3], None)
        .expect("second forward pass failed");
    println!("\nSecond pass:");
    println!("Fired neurons: {:?}", output2.fired_neurons);
    println!("Global step:   {}", output2.global_step);
}
