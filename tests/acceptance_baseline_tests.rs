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
        name: "acceptance_baseline_bimodal_ke",
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
    .with_solver(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45))
}

fn bimodal_data() -> Result<Data> {
    Ok(data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv")?)
}

#[test]
fn test_acceptance_baseline_npag_bimodal_ke() -> Result<()> {
    let result = EstimationProblem::builder(bimodal_ode_equation(), bimodal_data()?)
        .parameter(Parameter::bounded("ke", 0.001, 3.0))?
        .parameter(Parameter::bounded("v", 25.0, 250.0))?
        .method(Npag::new())
        .error(
            "1",
            AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
        )?
        .cycles(1000)
        .progress(false)
        .fit()?;
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
