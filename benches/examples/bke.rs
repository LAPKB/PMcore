use std::time::Duration;

use criterion::{black_box, criterion_group, Criterion};
use pmcore::prelude::*;
use pmcore::routines::settings::Settings;
use settings::read;

pub fn bke_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BKE Benchmarks");
    group.sample_size(10); // Set sample size to 2
    group.measurement_time(Duration::from_secs(600));

    group.bench_function("Analytical BKE", |b| {
        let eq = analytical_bke_equation();
        let settings = bke_settings();
        let data = data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv").unwrap();

        b.iter(|| {
            let mut algorithm =
                dispatch_algorithm(settings.clone(), eq.clone(), data.clone()).unwrap();
            let result = black_box(algorithm.fit());
            assert!(result.is_ok());
        });
    });

    group.bench_function("ODE BKE", |b| {
        let eq = ode_bke_equation();
        let settings = bke_settings();
        let data = data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv").unwrap();

        b.iter(|| {
            let mut algorithm =
                dispatch_algorithm(settings.clone(), eq.clone(), data.clone()).unwrap();
            let result = black_box(algorithm.fit());
            assert!(result.is_ok());
        });
    });

    group.finish();
}

criterion_group!(bke_group, bke_benchmarks);

fn bke_settings() -> Settings {
    read("examples/bimodal_ke/config.toml").unwrap()
}

fn analytical_bke_equation() -> equation::Analytical {
    // Your analytical BKE equation setup
    equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| {},
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    )
}

fn ode_bke_equation() -> equation::ODE {
    // Your ODE BKE equation setup
    equation::ODE::new(
        |x, p, _t, dx, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
        },
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    )
}
