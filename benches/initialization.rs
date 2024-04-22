use diol::{config::SampleCount, prelude::*};
use pmcore::routines::initialization::sobol::generate;

fn benchmark_sobol(bencher: Bencher, points: usize) {
    let range_params = vec![(0.0, 2.0), (0.0, 100.0)]; // Example parameter ranges
    let seed = 22;

    bencher.bench(|| {
        generate(points, &range_params, seed);
    });
}

fn main() -> std::io::Result<()> {
    // Configure benchmark
    let mut config = BenchConfig::default();
    config.sample_count = SampleCount(1000);
    config.output = Some(std::path::PathBuf::from("benches/initialization.json"));

    // Intialize bench
    let mut bench = Bench::new(config);

    // Register benchmark
    let points = [100, 1000, 10_000, 100_000, 1_000_000]; //
    bench.register(benchmark_sobol, points);

    // Run benchmarks
    bench.run()?;
    Ok(())
}
