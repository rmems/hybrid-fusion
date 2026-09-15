// SPDX-License-Identifier: MIT OR Apache-2.0

use hybrid_fusion::{
    HybridConfig, HybridNetwork, NeuroModulators, Result, SpikingNetwork, Tensor, Transformer,
};

struct DemoTransformer {
    dim: usize,
    max_seq: usize,
}

impl Transformer for DemoTransformer {
    fn hidden_states(&self, token_ids: &[u32]) -> Tensor {
        let seq = token_ids.len();
        let data: Vec<f32> = (0..seq * self.dim).map(|i| (i as f32) * 0.001).collect();
        Tensor::from_vec(data, &[seq, self.dim])
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq
    }
    fn param_count(&self) -> usize {
        self.dim * 12_000
    }
}

struct DemoSnn {
    channels: usize,
}

impl SpikingNetwork for DemoSnn {
    fn step(&mut self, stimuli: &[f32], _modulators: &NeuroModulators) -> Result<Vec<usize>> {
        Ok(stimuli
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > 0.0)
            .map(|(i, _)| i)
            .collect())
    }
    fn num_channels(&self) -> usize {
        self.channels
    }
}

fn main() -> hybrid_fusion::Result<()> {
    let cfg = HybridConfig::tiny();
    println!(
        "config | transformer.dim = {} | snn_channels = {} | lif = {} | izh = {}",
        cfg.transformer.dim, cfg.snn_input_channels, cfg.snn_lif_neurons, cfg.snn_izh_neurons,
    );

    let transformer = DemoTransformer {
        dim: cfg.transformer.dim,
        max_seq: cfg.transformer.max_seq_len,
    };
    let snn = DemoSnn {
        channels: cfg.snn_input_channels,
    };
    let mut net = HybridNetwork::try_new(transformer, snn, cfg)?;
    println!(
        "HybridNetwork ready | transformer params ~ {} | snn channels = {}",
        net.transformer.param_count(),
        net.snn.num_channels(),
    );

    let n_steps = 10usize;
    let token_ids: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    println!(
        "\nRunning {n_steps} steps on {} token prompt ...",
        token_ids.len()
    );
    println!(
        "{:>5}  {:>10}  {:>12}  {:>9}",
        "step", "stim[0]", "fired_count", "tempC"
    );
    println!("{}", "-".repeat(48));

    for t in 0..n_steps {
        let gpu_temp = 60.0 + t as f32 * 1.5;

        let modulators = NeuroModulators {
            dopamine: (gpu_temp / 100.0).clamp(0.0, 1.0),
            cortisol: ((gpu_temp - 60.0) / 40.0).clamp(0.0, 1.0),
            acetylcholine: 0.7,
            tempo: 1.0,
            aux_dopamine: 0.5,
        };

        let out = net.forward(&token_ids, Some(modulators))?;

        println!(
            "{:>5}  {:>10.4}  {:>12}  {:>6.1}C",
            out.global_step,
            out.stimuli[0],
            out.fired_neurons.len(),
            gpu_temp,
        );
    }

    println!("{}", "-".repeat(48));
    println!(
        "\nfinal global_step = {} | embedding width = {}",
        net.global_step(),
        net.config().transformer.dim,
    );
    println!("Done! Pure transformer -> projector -> SNN loop complete.");
    Ok(())
}
