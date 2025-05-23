use anyhow::Result;
use criterion::{criterion_group, criterion_main, Criterion};
use pmcore::prelude::*;

use std::hint::black_box;
fn create_equation() -> equation::ODE {
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

fn setup_simulation() -> Result<(Settings, equation::ODE, data::Data)> {
    let params = Parameters::new()
        .add("ke", 0.001, 3.0)
        .add("v", 25.0, 250.0);

    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_model(ErrorType::Additive, 0.0, (0.0, 0.5, 0.0, 0.0))
        .build();

    settings.set_cycles(1000);
    settings.set_prior(Prior::sobol(2048, 22));
    settings.set_output_path("examples/bimodal_ke/output");

    let data = data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv")?;
    Ok((settings, create_equation(), data))
}

fn benchmark_bimodal_ke(c: &mut Criterion) {
    let (settings, eq, data) = setup_simulation().unwrap();

    c.bench_function("bimodal_ke_fit", |b| {
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

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = benchmark_bimodal_ke
}
criterion_main!(benches);
