use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use faer::Mat;
use pmcore::routines::evaluation::clarabel::clarabel_weights;
use pmcore::routines::evaluation::em::em_weights;
use pmcore::routines::evaluation::frank_wolfe::frank_wolfe_weights;
use pmcore::routines::evaluation::ipm::burke;
use pmcore::routines::evaluation::lbfgs::lbfgs_weights_default;
use pmcore::routines::evaluation::mirror_descent::mirror_descent_weights;
use pmcore::routines::evaluation::pgd::pgd_weights;
use pmcore::routines::evaluation::squarem::squarem_weights;
use pmcore::routines::evaluation::stochastic_mirror_descent::stochastic_mirror_descent_weights;
use pmcore::structs::psi::Psi;

fn make_psi(n_sub: usize, n_point: usize) -> Psi {
    let mat = Mat::from_fn(n_sub, n_point, |i, j| {
        if j == 0 {
            10.0 + (n_sub as f64) * 0.1
        } else {
            1.0 + 0.01 * (i as f64) + 0.005 * (j as f64)
        }
    });
    Psi::from(mat)
}

fn bench_weights(c: &mut Criterion) {
    let sizes = [(10, 10), (100, 100), (500, 500), (1000, 1000)];
    let algos: &[(
        &str,
        fn(&Psi) -> anyhow::Result<(pmcore::structs::weights::Weights, f64)>,
    )] = &[
        ("burke", burke),
        ("clarabel", clarabel_weights),
        ("em", em_weights),
        ("squarem", squarem_weights),
        ("lbfgs", lbfgs_weights_default),
        ("mirror_descent", mirror_descent_weights),
        ("pgd", pgd_weights),
        ("frank_wolfe", frank_wolfe_weights),
        (
            "stochastic_mirror_descent",
            stochastic_mirror_descent_weights,
        ),
    ];
    for &(name, algo) in algos {
        let mut group = c.benchmark_group(format!("weights_{}", name));
        for &(n_sub, n_point) in &sizes {
            let psi = make_psi(n_sub, n_point);
            group.throughput(Throughput::Elements((n_sub * n_point) as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}x{}", n_sub, n_point)),
                &psi,
                |b, psi| {
                    b.iter(|| {
                        let _ = algo(psi).unwrap();
                    });
                },
            );
        }
        group.finish();
    }
}

use std::time::Duration;

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .measurement_time(Duration::from_secs(5));
    targets = bench_weights
}
criterion_main!(benches);
