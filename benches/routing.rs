// SPDX-License-Identifier: MIT OR Apache-2.0

//! Criterion benchmarks for [`hybrid_fusion::top_k_indices`] and
//! [`hybrid_fusion::softmax`] at several expert counts (RM-1470).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hybrid_fusion::{softmax, top_k_indices};

fn bench_scores(n: usize, seed: u64) -> Vec<f32> {
    let mut st = seed;
    (0..n)
        .map(|i| {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (st >> 33) as f32 / (1u64 << 31) as f32;
            ((i as f32 * 0.001) + u).sin()
        })
        .collect()
}

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");
    for &n in &[128usize, 4_096, 65_536, 100_000, 1_000_000] {
        let scores = bench_scores(n, 0x5EED);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &scores, |b, scores| {
            b.iter(|| softmax(scores).unwrap());
        });
    }
    group.finish();
}

fn bench_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("top_k_indices");
    const TOP_K: usize = 8;
    for &n in &[128usize, 4_096, 65_536, 100_000, 1_000_000] {
        let weights = softmax(&bench_scores(n, 0xC0FF)).unwrap();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &weights, |b, weights| {
            b.iter(|| top_k_indices(weights, TOP_K));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_softmax, bench_top_k);
criterion_main!(benches);
