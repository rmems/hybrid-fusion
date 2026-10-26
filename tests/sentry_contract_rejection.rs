// SPDX-License-Identifier: MIT OR Apache-2.0

//! RM-1443: a host-side rejection of the backend's fired set is a contract
//! violation, not a backend/runtime failure, so it must not reach Sentry.
//!
//! A counting transport is bound to a thread-local hub. The positive control
//! (a backend `Err`) proves the harness would see a capture if one happened.
#![cfg(feature = "sentry")]

use hybrid_fusion::{
    HybridConfig, HybridError, HybridNetwork, NeuroModulators, SpikingNetwork, Tensor, Transformer,
};
use sentry::{ClientOptions, Envelope, Hub, Transport, TransportFactory};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct T {
    dim: usize,
    max_seq: usize,
}

impl Transformer for T {
    fn hidden_states(&self, token_ids: &[u32]) -> Tensor {
        let n = token_ids.len() * self.dim;
        Tensor::from_vec(vec![0.1; n], &[token_ids.len(), self.dim])
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq
    }
    fn param_count(&self) -> usize {
        1
    }
}

struct S {
    outcome: Result<Vec<usize>, String>,
}

impl SpikingNetwork for S {
    fn num_channels(&self) -> usize {
        8
    }
    fn num_neurons(&self) -> usize {
        3
    }
    fn step(&mut self, _: &[f32], _: &NeuroModulators) -> hybrid_fusion::Result<Vec<usize>> {
        self.outcome.clone().map_err(HybridError::SnnStep)
    }
}

struct CountingTransport(Arc<AtomicUsize>);

impl Transport for CountingTransport {
    fn send_envelope(&self, _: Envelope) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

struct Factory(Arc<AtomicUsize>);

impl TransportFactory for Factory {
    fn create_transport(&self, _: &ClientOptions) -> Arc<dyn Transport> {
        Arc::new(CountingTransport(self.0.clone()))
    }
}

/// Run one `forward` against a backend outcome with a capturing hub bound and
/// return `(error, envelopes_sent)`.
fn run(outcome: Result<Vec<usize>, String>) -> (Option<HybridError>, usize) {
    let sent = Arc::new(AtomicUsize::new(0));
    let client = Arc::new(sentry::Client::from((
        "https://public@example.invalid/1",
        ClientOptions {
            transport: Some(Arc::new(Factory(sent.clone()))),
            ..Default::default()
        },
    )));
    let hub = Arc::new(Hub::new(
        Some(client.clone()),
        Arc::new(sentry::Scope::default()),
    ));
    let mut cfg = HybridConfig::tiny();
    cfg.snn_input_channels = 8;
    let t = T {
        dim: cfg.transformer.dim,
        max_seq: cfg.transformer.max_seq_len,
    };
    let mut net = HybridNetwork::try_new(t, S { outcome }, cfg).unwrap();
    let err = Hub::run(hub, || net.forward(&[1], None).err());
    client.flush(None);
    (err, sent.load(Ordering::SeqCst))
}

#[test]
fn backend_error_is_captured_positive_control() {
    let (err, sent) = run(Err("boom".into()));
    assert!(matches!(err, Some(HybridError::SnnStep(_))));
    assert_eq!(sent, 1, "harness must observe a backend failure capture");
}

#[test]
fn contract_rejections_are_not_captured() {
    let (err, sent) = run(Ok(vec![3]));
    assert!(matches!(
        err,
        Some(HybridError::SnnSpikeOutOfRange {
            neuron: 3,
            num_neurons: 3
        })
    ));
    assert_eq!(sent, 0, "out-of-range spike must not reach Sentry");

    let (err, sent) = run(Ok(vec![0, 0]));
    assert!(matches!(
        err,
        Some(HybridError::SnnDuplicateSpike { neuron: 0 })
    ));
    assert_eq!(sent, 0, "duplicate spike must not reach Sentry");

    let (err, sent) = run(Ok(vec![2, 0, 1]));
    assert!(err.is_none());
    assert_eq!(sent, 0);
}
