use anyhow::Result;
use criterion::{criterion_group, criterion_main, Criterion};
use pmcore::prelude::*;
use settings::{Parameters, Settings};

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
    let mut settings = Settings::new();
    let params = Parameters::builder()
        .add("ke", 0.001, 3.0, true)
        .add("v", 25.0, 250.0, true)
        .build()?;

    settings.set_parameters(params);
    settings.set_cycles(1000);
    settings.set_error_poly((0.0, 0.5, 0.0, 0.0));
    settings.set_error_type(ErrorType::Add);
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
                algorithm.fit().unwrap()
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
