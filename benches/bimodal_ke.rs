use anyhow::Result;
use criterion::{criterion_group, criterion_main, Criterion};
use pmcore::prelude::*;

use std::hint::black_box;

fn create_equation() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[1] + b[1];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[1] = x[0] / v;
        },
    )
}

fn create_parameter_space() -> ParameterSpace {
    ParameterSpace::new()
        .add(ParameterSpec::bounded("ke", 0.001, 3.0))
        .add(ParameterSpec::bounded("v", 25.0, 250.0))
}

fn create_error_models() -> Result<AssayErrorModels> {
    Ok(AssayErrorModels::new().add(
        1,
        AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
    )?)
}

fn load_data() -> Result<data::Data> {
    Ok(data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv")?)
}

fn setup_with_algorithm(method: NonparametricMethod) -> Result<EstimationProblem<equation::ODE>> {
    let ems = create_error_models()?;
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(1, "cp"))
        .with_assay_error_models(ems);
    let model = ModelDefinition::builder(create_equation())
        .parameters(create_parameter_space())
        .observations(observations)
        .build()?;
    let data = load_data()?;
    EstimationProblem::builder(model, data)
        .method(EstimationMethod::Nonparametric(method))
        .output(OutputPlan::disabled())
        .runtime(RuntimeOptions {
            cycles: 1000,
            progress: false,
            prior: Some(Prior::sobol(2048, 22)),
            ..RuntimeOptions::default()
        })
        .build()
}

fn setup_npag() -> Result<EstimationProblem<equation::ODE>> {
    setup_with_algorithm(NonparametricMethod::Npag(NpagOptions))
}

fn setup_npod() -> Result<EstimationProblem<equation::ODE>> {
    setup_with_algorithm(NonparametricMethod::Npod(NpodOptions))
}

fn setup_postprob() -> Result<EstimationProblem<equation::ODE>> {
    setup_with_algorithm(NonparametricMethod::Postprob(PostProbOptions))
}

fn benchmark_algorithm<F>(c: &mut Criterion, bench_name: &str, setup_fn: F)
where
    F: Fn() -> Result<EstimationProblem<equation::ODE>>,
{
    let problem = setup_fn().unwrap();

    c.bench_function(bench_name, |b| {
        b.iter_with_setup(
            || problem.clone(),
            |problem| black_box(problem.run().unwrap()),
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
