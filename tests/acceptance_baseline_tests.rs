use anyhow::Result;
use pmcore::{model::BoundedParameter, prelude::*};

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
    let result = EstimationProblem::builder(bimodal_ode_equation(), bimodal_data().unwrap())
        .nonparametric()
        .parameter(BoundedParameter::new("ke", 0.1, 1.0))
        .parameter(BoundedParameter::new("v", 1.0, 20.0))
        .error(
            "0",
            AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 2.0),
        )
        .build()?
        .fit_with(NpagConfig::default())?;

    // This is the canonical rewrite-blocking nonparametric baseline for the bimodal_ke example.
    assert_close(result.objf(), -425.60904902364695, 1e-6, "npag.objf");
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
