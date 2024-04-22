use diol::{config::SampleCount, prelude::*};
use ndarray::Array1;
use pmcore::routines::evaluation::prob::normal_likelihood;

fn benchmark_normal_likelihood(bencher: Bencher, size: usize) {
    // Generate sample data for benchmarking
    let ypred = Array1::from_vec((0..size).map(|x| x as f64).collect());
    let yobs = Array1::from_vec((0..size).map(|x| x as f64 * 1.1).collect());
    let sigma = Array1::from_elem(size, 1.0);

    bencher.bench(|| {
        normal_likelihood(&ypred, &yobs, &sigma);
    });
}

fn main() -> std::io::Result<()> {
    // Configure benchmark
    let mut config = BenchConfig::default();
    config.sample_count = SampleCount(1000);
    config.output = Some(std::path::PathBuf::from("benches/evaluation.json"));

    // Intialize bench
    let mut bench = Bench::new(config);

    // Register benchmark
    let sizes = [100, 1000, 10_000, 100_000, 1_000_000]; //
    bench.register(benchmark_normal_likelihood, sizes);

    // Run benchmarks
    bench.run()?;
    Ok(())
}
