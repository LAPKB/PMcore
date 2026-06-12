use anyhow::Result;
use criterion::{criterion_group, criterion_main, Criterion};
use pmcore::{
    algorithms::{Algorithm, Fitter},
    prelude::*,
};

use std::hint::black_box;

fn create_equation() -> equation::ODE {
    ode! {
        name: "bimodal_ke",
        params: [ke, v],
        states: [central],
        outputs: [outeq_1],
        routes: [
            infusion(input_1) -> central,
        ],
        diffeq: |x, _t, dx| {
            dx[central] = -ke * x[central];
        },
        out: |x, _t, y| {
            y[outeq_1] = x[central] / v;
        },
    }
}

fn load_data() -> Result<data::Data> {
    Ok(data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv")?)
}

fn setup_npag() -> Result<EstimationProblem<equation::ODE, NonParametric>> {
    let data = load_data()?;
    EstimationProblem::builder(create_equation(), data)
        .nonparametric()
        .parameter(Parameter::bounded("ke", 0.001, 3.0))
        .parameter(Parameter::bounded("v", 25.0, 250.0))
        .error_model(
            "outeq_1",
            AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
        )
        .build()
}

fn setup_npod() -> Result<EstimationProblem<equation::ODE, NonParametric>> {
    let data = load_data()?;
    EstimationProblem::builder(create_equation(), data)
        .nonparametric()
        .parameter(Parameter::bounded("ke", 0.001, 3.0))
        .parameter(Parameter::bounded("v", 25.0, 250.0))
        .error_model(
            "outeq_1",
            AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
        )
        .build()
}

fn setup_postprob() -> Result<EstimationProblem<equation::ODE, NonParametric>> {
    let data = load_data()?;
    EstimationProblem::builder(create_equation(), data)
        .nonparametric()
        .parameter(Parameter::bounded("ke", 0.001, 3.0))
        .parameter(Parameter::bounded("v", 25.0, 250.0))
        .error_model(
            "outeq_1",
            AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
        )
        .build()
}

fn benchmark_algorithm<F, A>(c: &mut Criterion, bench_name: &str, setup_fn: F, config: A)
where
    F: Fn() -> Result<EstimationProblem<equation::ODE, NonParametric>>,
    A: Algorithm<equation::ODE, NonParametric> + Clone,
    A::Runner: Fitter<equation::ODE>,
{
    c.bench_function(bench_name, |b| {
        b.iter_with_setup(
            || setup_fn().unwrap(),
            |problem| black_box(problem.fit_with(config.clone()).unwrap()),
        )
    });
}

fn benchmark_bimodal_ke_npag(c: &mut Criterion) {
    benchmark_algorithm(c, "bimodal_ke_npag", setup_npag, NpagConfig::default());
}

fn benchmark_bimodal_ke_npod(c: &mut Criterion) {
    benchmark_algorithm(c, "bimodal_ke_npod", setup_npod, NpodConfig::default());
}

fn benchmark_bimodal_ke_postprob(c: &mut Criterion) {
    benchmark_algorithm(
        c,
        "bimodal_ke_postprob",
        setup_postprob,
        NpmapConfig::default(),
    );
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = benchmark_bimodal_ke_npag, benchmark_bimodal_ke_npod, benchmark_bimodal_ke_postprob
}
criterion_main!(benches);

