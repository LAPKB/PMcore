use criterion::{black_box, criterion_group, Criterion};
use pmcore::prelude::*;
use pmcore::routines::settings::Settings;
use settings::read;
use std::time::Duration;

pub fn tel_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("TEL Benchmarks");
    group.sample_size(10); // Set sample size to 2
    group.measurement_time(Duration::from_secs(1));

    group.bench_function("Analytical TEL", |b| {
        let eq = analytical_tel_equation();
        let settings = tel_settings();
        let data = data::read_pmetrics("examples/two_eq_lag/two_eq_lag.csv").unwrap();

        b.iter(|| {
            let mut algorithm =
                dispatch_algorithm(settings.clone(), eq.clone(), data.clone()).unwrap();
            let result = black_box(algorithm.fit());
            assert!(result.is_ok());
        });
    });

    group.bench_function("ODE TEL", |b| {
        let eq = ode_tel_equation();
        let settings = tel_settings();
        let data = data::read_pmetrics("examples/two_eq_lag/two_eq_lag.csv").unwrap();

        b.iter(|| {
            let mut algorithm =
                dispatch_algorithm(settings.clone(), eq.clone(), data.clone()).unwrap();
            let result = black_box(algorithm.fit());
            assert!(result.is_ok());
        });
    });

    group.finish();
}

criterion_group!(tel_group, tel_benchmarks);

fn tel_settings() -> Settings {
    read("examples/two_eq_lag/config.toml").unwrap()
}

fn analytical_tel_equation() -> equation::Analytical {
    // Your analytical TEL equation setup
    equation::Analytical::new(
        one_compartment_with_absorption,
        |_p, _t, _cov| {},
        |p| {
            fetch_params!(p, _ka, _ke, tlag, _v);
            lag! {0 => tlag}
        },
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _tlag, v);
            y[0] = x[1] / v;
        },
        (2, 1),
    )
}

fn ode_tel_equation() -> equation::ODE {
    // Your ODE TEL equation setup
    equation::ODE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            fetch_params!(p, ka, ke, _tlag, _v);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1];
        },
        |p| {
            fetch_params!(p, _ka, _ke, tlag, _v);
            lag! {0 => tlag}
        },
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _tlag, v);
            y[0] = x[1] / v;
        },
        (2, 1),
    )
}
