use anyhow::Result;
use criterion::{criterion_group, criterion_main, Criterion};
use pmcore::prelude::*;

use std::hint::black_box;

fn create_equation() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0] + b[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    )
}

fn create_parameters() -> Parameters {
    Parameters::new()
        .add("ke", 0.001, 3.0)
        .add("v", 25.0, 250.0)
}

fn create_error_models() -> Result<ErrorModels> {
    Ok(ErrorModels::new().add(
        0,
        ErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
    )?)
}

fn load_data() -> Result<data::Data> {
    Ok(data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv")?)
}

fn setup_with_algorithm(algorithm: Algorithm) -> Result<(Settings, equation::ODE, data::Data)> {
    let params = create_parameters();
    let ems = create_error_models()?;

    let mut settings = Settings::builder()
        .set_algorithm(algorithm)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    settings.set_cycles(1000);
    settings.set_prior(Prior::sobol(2048, 22));
    settings.disable_output();
    settings.set_progress(false);

    let data = load_data()?;
    Ok((settings, create_equation(), data))
}

fn setup_npag() -> Result<(Settings, equation::ODE, data::Data)> {
    setup_with_algorithm(Algorithm::NPAG)
}

fn setup_npod() -> Result<(Settings, equation::ODE, data::Data)> {
    setup_with_algorithm(Algorithm::NPOD)
}

fn setup_postprob() -> Result<(Settings, equation::ODE, data::Data)> {
    setup_with_algorithm(Algorithm::POSTPROB)
}

fn benchmark_algorithm<F>(c: &mut Criterion, bench_name: &str, setup_fn: F)
where
    F: Fn() -> Result<(Settings, equation::ODE, data::Data)>,
{
    let (settings, eq, data) = setup_fn().unwrap();

    c.bench_function(bench_name, |b| {
        b.iter_with_setup(
            || (settings.clone(), eq.clone(), data.clone()),
            |(s, e, d)| {
                let mut algorithm = dispatch_algorithm(s, e, d).unwrap();
                let result = algorithm.fit().unwrap();
                black_box(result)
            },
        )
    });
}

fn benchmark_bimodal_ke_npag(c: &mut Criterion) {
    benchmark_algorithm(c, "bimodal_ke_npag", setup_npag);
}

fn benchmark_bimodal_ke_npod(c: &mut Criterion) {
    benchmark_algorithm(c, "bimodal_ke_npod", setup_npod);
}

fn benchmark_bimodal_ke_postprob(c: &mut Criterion) {
    benchmark_algorithm(c, "bimodal_ke_postprob", setup_postprob);
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = benchmark_bimodal_ke_npag, benchmark_bimodal_ke_npod, benchmark_bimodal_ke_postprob
}
criterion_main!(benches);
