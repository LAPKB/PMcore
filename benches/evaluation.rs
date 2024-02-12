use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;
use pmcore::routines::evaluation::prob::normal_likelihood;

fn benchmark_normal_likelihood(c: &mut Criterion) {
    let sizes = [100, 1000, 10_000, 100_000, 1_000_000]; // Example sizes to benchmark

    for &size in &sizes {
        // Generate sample data for benchmarking
        let ypred = Array1::from_vec((0..size).map(|x| x as f64).collect());
        let yobs = Array1::from_vec((0..size).map(|x| x as f64 * 1.1).collect());
        let sigma = Array1::from_elem(size, 1.0);

        c.bench_function(&format!("normal_likelihood_{size}"), |b| {
            b.iter(|| {
                normal_likelihood(
                    black_box(&ypred),
                    black_box(&yobs),
                    black_box(&sigma)
                )
            });
        });
    }
}

criterion_group!(benches, benchmark_normal_likelihood);
criterion_main!(benches);
