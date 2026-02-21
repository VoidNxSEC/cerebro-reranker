//! Benchmarks for scoring performance

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use cerebro_scorer::*;

fn generate_embeddings(n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| (0..dim).map(|_| rand::random::<f32>()).collect())
        .collect()
}

fn bench_scoring(c: &mut Criterion) {
    let mut group = c.benchmark_group("scoring");

    for dim in [128, 384, 768, 1024] {
        let query = (0..dim).map(|_| rand::random::<f32>()).collect::<Vec<_>>();
        let docs = generate_embeddings(100, dim);

        group.bench_with_input(
            BenchmarkId::new("batch_score", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    batch_cosine_similarity(
                        black_box(&query),
                        black_box(&docs)
                    ).unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("top_k");

    for size in [100, 1000, 10000] {
        let scores: Vec<f32> = (0..size).map(|_| rand::random()).collect();

        group.bench_with_input(
            BenchmarkId::new("top_k_selection", size),
            &size,
            |b, _| {
                b.iter(|| {
                    top_k_indices(black_box(&scores), black_box(10))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_scoring, bench_top_k);
criterion_main!(benches);
