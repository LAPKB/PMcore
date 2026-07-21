use criterion::{criterion_group, criterion_main, Criterion};
use pmcore::prelude::*;

fn model() -> pharmsol::equation::Analytical {
    analytical! {
        name: "saem_v01_benchmark",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [infusion(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    }
}

fn data() -> Data {
    Data::new(
        [
            ("s1", 100.0, [4.70, 4.15, 3.15, 1.75]),
            ("s2", 120.0, [4.65, 4.25, 3.45, 2.15]),
            ("s3", 80.0, [4.45, 3.75, 2.65, 1.20]),
            ("s4", 110.0, [4.55, 4.10, 3.25, 1.95]),
        ]
        .into_iter()
        .map(|(id, dose, observations)| {
            Subject::builder(id)
                .infusion(0.0, dose, "iv", 0.5)
                .observation(0.5, observations[0], "cp")
                .observation(1.0, observations[1], "cp")
                .observation(2.0, observations[2], "cp")
                .observation(4.0, observations[3], "cp")
                .build()
        })
        .collect(),
    )
}

fn problem() -> Result<EstimationProblem<pharmsol::equation::Analytical, Parametric>> {
    EstimationProblem::parametric(model(), data())
        .parameter(Parameter::log("ke").with_initial(0.30))
        .parameter(Parameter::log("v").with_initial(20.0))
        .omega(Omega::diagonal([("ke", 0.09), ("v", 0.09)]))
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.25)).fixed(),
        )
        .build()
}

fn config(compute_map: bool) -> SaemConfig {
    SaemConfig::new()
        .seed(20_260_710)
        .n_chains(3)
        .mcmc_iterations(2)
        .burn_in(2)
        .k1_iterations(8)
        .k2_iterations(4)
        .compute_map(compute_map)
        .map_max_iterations(100)
}

fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("saem_v01");
    group.sample_size(20);
    group.bench_function("core_12_cycles", |b| {
        b.iter(|| {
            let problem = match problem() {
                Ok(problem) => problem,
                Err(error) => panic!("V01 benchmark problem failed: {error}"),
            };
            match problem.fit_with(config(false)) {
                Ok(result) => result,
                Err(error) => panic!("V01 core fit failed: {error}"),
            }
        });
    });
    group.bench_function("core_plus_conditional_modes", |b| {
        b.iter(|| {
            let problem = match problem() {
                Ok(problem) => problem,
                Err(error) => panic!("V01 benchmark problem failed: {error}"),
            };
            match problem.fit_with(config(true)) {
                Ok(result) => result,
                Err(error) => panic!("V01 fit with modes failed: {error}"),
            }
        });
    });
    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
