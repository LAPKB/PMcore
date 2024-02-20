use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pmcore::routines::initialization::sobol::generate; // Import the function you want to benchmark

fn benchmark_sobol(c: &mut Criterion) {
    let range_params = vec![(0.0, 2.0), (0.0, 100.0)]; // Example parameter ranges
    let seed = 22;

    c.bench_function("generate_1_000", |b| {
        b.iter(|| {
            // Benchmarking with 1000 points
            let _ = generate(black_box(1000), black_box(&range_params), black_box(seed));
        });
    });

    c.bench_function("generate_10_000", |b| {
        b.iter(|| {
            // Benchmarking with 10,000 points
            let _ = generate(black_box(10000), black_box(&range_params), black_box(seed));
        });
    });

    c.bench_function("generate_100_000", |b| {
        b.iter(|| {
            // Benchmarking with 100,000 points
            let _ = generate(black_box(100000), black_box(&range_params), black_box(seed));
        });
    });
}

criterion_group!(benches, benchmark_sobol);
criterion_main!(benches);
