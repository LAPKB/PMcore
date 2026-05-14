use anyhow::Result;
use criterion::{criterion_group, criterion_main, Criterion};
use pmcore::prelude::*;

use std::hint::black_box;

fn create_equation() -> equation::ODE {
    ode! {
        name: "bimodal_ke",
        params: [ke, v],
        states: [central],
        outputs: [1],
        routes: [
            infusion(1) -> central,
        ],
        diffeq: |x, _t, dx| {
            dx[central] = -ke * x[central];
        },
        out: |x, _t, y| {
            y[1] = x[central] / v;
        },
    }
}

fn create_error_model() -> AssayErrorModel {
    AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0)
}

fn load_data() -> Result<data::Data> {
    Ok(data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv")?)
}

fn setup_npag() -> Result<EstimationProblem<equation::ODE>> {
    let data = load_data()?;
    EstimationProblem::builder(create_equation(), data)
        .parameter(Parameter::bounded("ke", 0.001, 3.0))?
        .parameter(Parameter::bounded("v", 25.0, 250.0))?
        .method(Npag::new())
        .error("1", create_error_model())?
        .cycles(1000)
        .progress(false)
        .prior(Prior::sobol(2048, 22))
        .build()
}

fn setup_npod() -> Result<EstimationProblem<equation::ODE>> {
    let data = load_data()?;
    EstimationProblem::builder(create_equation(), data)
        .parameter(Parameter::bounded("ke", 0.001, 3.0))?
        .parameter(Parameter::bounded("v", 25.0, 250.0))?
        .method(Npod::new())
        .error("1", create_error_model())?
        .cycles(1000)
        .progress(false)
        .prior(Prior::sobol(2048, 22))
        .build()
}

fn setup_postprob() -> Result<EstimationProblem<equation::ODE>> {
    let data = load_data()?;
    EstimationProblem::builder(create_equation(), data)
        .parameter(Parameter::bounded("ke", 0.001, 3.0))?
        .parameter(Parameter::bounded("v", 25.0, 250.0))?
        .method(PostProb::new())
        .error("1", create_error_model())?
        .cycles(1000)
        .progress(false)
        .prior(Prior::sobol(2048, 22))
        .build()
}

fn benchmark_algorithm<F>(c: &mut Criterion, bench_name: &str, setup_fn: F)
where
    F: Fn() -> Result<EstimationProblem<equation::ODE>>,
{
    let problem = setup_fn().unwrap();

    c.bench_function(bench_name, |b| {
        b.iter_with_setup(
            || problem.clone(),
            |problem| black_box(problem.fit().unwrap()),
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
