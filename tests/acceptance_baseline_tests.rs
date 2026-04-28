use anyhow::Result;
use pmcore::prelude::*;

fn assert_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    let difference = (actual - expected).abs();
    assert!(
        difference <= tolerance,
        "{label}: expected {expected}, got {actual}, abs diff {difference} > {tolerance}"
    );
}

fn bimodal_ode_equation() -> equation::ODE {
    ode! {
        diffeq: |x, p, _t, dx, b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[1] + b[1];
        },
        out: |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[1] = x[0] / v;
        },
    }
    .with_solver(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45))
}

fn bimodal_data() -> Result<Data> {
    Ok(data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv")?)
}

fn bimodal_npag_model() -> Result<ModelDefinition<equation::ODE>> {
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(1, "cp"))
        .with_assay_error_models(AssayErrorModels::new().add(
            1,
            AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
        )?);

    ModelDefinition::builder(bimodal_ode_equation())
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ke", 0.001, 3.0))
                .add(ParameterSpec::bounded("v", 25.0, 250.0)),
        )
        .observations(observations)
        .build()
}

#[test]
fn test_acceptance_baseline_npag_bimodal_ke() -> Result<()> {
    let result = EstimationProblem::builder(bimodal_npag_model()?, bimodal_data()?)
        .method(EstimationMethod::Nonparametric(NonparametricMethod::Npag(
            NpagOptions::default(),
        )))
        .output(OutputPlan::disabled())
        .runtime(RuntimeOptions {
            cycles: 1000,
            progress: false,
            ..RuntimeOptions::default()
        })
        .run()?;
    let summary = result.summary();
    let population = result.population_summary();
    let result = result
        .as_nonparametric()
        .expect("NPAG acceptance baseline should yield a nonparametric result");

    // This is the canonical rewrite-blocking nonparametric baseline for the bimodal_ke example.
    assert_close(
        summary.objective_function,
        -425.60904902364695,
        1e-6,
        "npag.objf",
    );
    assert!(summary.converged);
    assert_eq!(summary.iterations, 288);
    assert_eq!(result.get_theta().nspp(), 46);
    assert_close(
        population.parameters[0].mean,
        0.187047284678325,
        1e-6,
        "npag.ke.mean",
    );
    assert_close(
        population.parameters[1].mean,
        107.94241284196241,
        1e-6,
        "npag.v.mean",
    );
    Ok(())
}
