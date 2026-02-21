use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use scorer::scorer;

fn random_vec(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");

    for dim in [384, 768, 1024] {
        let a = random_vec(dim);
        let b = random_vec(dim);

        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |bench, _| {
                bench.iter(|| scorer::cosine_similarity(black_box(&a), black_box(&b)));
            },
        );
    }

    group.finish();
}

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for dim in [384, 768, 1024] {
        let a = random_vec(dim);
        let b = random_vec(dim);

        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |bench, _| {
                bench.iter(|| scorer::dot_product(black_box(&a), black_box(&b)));
            },
        );
    }

    group.finish();
}

fn bench_score_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("score_batch");

    let dim = 384; // MiniLM embedding size

    for batch_size in [32, 128, 512] {
        let queries: Vec<Vec<f32>> = (0..batch_size).map(|_| random_vec(dim)).collect();
        let docs: Vec<Vec<f32>> = (0..batch_size).map(|_| random_vec(dim)).collect();

        let q_refs: Vec<&[f32]> = queries.iter().map(|v| v.as_slice()).collect();
        let d_refs: Vec<&[f32]> = docs.iter().map(|v| v.as_slice()).collect();

        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |bench, _| {
                bench.iter(|| scorer::score_batch(black_box(&q_refs), black_box(&d_refs)));
            },
        );
    }

    group.finish();
}

fn bench_normalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize");

    for dim in [384, 768] {
        let original = random_vec(dim);

        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |bench, _| {
                let mut v = original.clone();
                bench.iter(|| {
                    v.copy_from_slice(&original);
                    scorer::normalize(black_box(&mut v));
                });
            },
        );
    }

    group.finish();
}

fn bench_cache_throughput(c: &mut Criterion) {
    use scorer::cache::TinyLfuCache;
    use std::sync::Arc;

    let mut group = c.benchmark_group("cache");

    let cache = Arc::new(TinyLfuCache::new(10_000));

    // Pre-populate
    for i in 0..5000 {
        cache.put(format!("key:{i}"), format!("value:{i}"), None);
    }

    group.bench_function("get_hit", |bench| {
        let mut i = 0u64;
        bench.iter(|| {
            let key = format!("key:{}", i % 5000);
            black_box(cache.get(&key));
            i += 1;
        });
    });

    group.bench_function("get_miss", |bench| {
        let mut i = 0u64;
        bench.iter(|| {
            let key = format!("miss:{i}");
            black_box(cache.get(&key));
            i += 1;
        });
    });

    group.bench_function("put", |bench| {
        let mut i = 5000u64;
        bench.iter(|| {
            cache.put(format!("new:{i}"), format!("v:{i}"), None);
            i += 1;
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_cosine_similarity,
    bench_dot_product,
    bench_score_batch,
    bench_normalize,
    bench_cache_throughput,
);
criterion_main!(benches);
