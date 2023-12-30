use criterion::{black_box, criterion_group, criterion_main, Criterion};

/// Benchmark the Sobol initialization routine using 1000 points in 10 dimensions
fn benchmark_sobol(c: &mut Criterion) {
    c.bench_function("sobol", |b| {
        b.iter(|| {
            let _ = npcore::routines::initialization::sobol::generate(
                black_box(1000),
                black_box(&vec![(0.0, 1.0); 10]),
                black_box(22),
            );
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .measurement_time(std::time::Duration::from_secs(10)) // Measure for 10 seconds
        .noise_threshold(0.10); // Performance changes less than 10% will be ignored
    targets = benchmark_sobol
}
criterion_main!(benches);
