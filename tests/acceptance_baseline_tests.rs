use anyhow::Result;
use pharmsol::{ResidualErrorModel, ResidualErrorModels};
use pmcore::prelude::*;

#[allow(dead_code)]
#[path = "saem_validation/reference.rs"]
mod saem_reference;

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

fn simple_focei_equation() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke);
            dx[0] = -ke * x[0] + b[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, v);
            y[0] = x[0] / v;
        },
    )
}

fn bimodal_analytical_equation() -> equation::Analytical {
    equation::Analytical::new(
        |x, p, t, rateiv, _cov| {
            let mut xout = x.clone();
            fetch_params!(p, ke, _v);
            xout[0] = x[0] * (-ke * t).exp() + rateiv[1] / ke * (1.0 - (-ke * t).exp());
            xout
        },
        |_p, _t, _cov| {},
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[1] = x[0] / v;
        },
    )
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

fn bimodal_saem_problem() -> Result<EstimationProblem<equation::Analytical>> {
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(1, "cp"))
        .with_residual_error_models(
            ResidualErrorModels::new().add(1, ResidualErrorModel::proportional(0.1)),
        );

    let model = ModelDefinition::builder(bimodal_analytical_equation())
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ke", 0.01, 0.5))
                .add(ParameterSpec::bounded("v", 50.0, 180.0)),
        )
        .observations(observations)
        .build()?;

    EstimationProblem::builder(model, bimodal_data()?)
        .method(EstimationMethod::Parametric(ParametricMethod::Saem(
            SaemOptions::default(),
        )))
        .output(OutputPlan::disabled())
        .runtime(RuntimeOptions {
            progress: false,
            ..RuntimeOptions::default()
        })
        .build()
}

fn canonical_focei_data() -> Data {
    Subject::builder("1")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 10.0, 0)
        .observation(2.0, 8.0, 0)
        .build()
        .into()
}

fn canonical_focei_model() -> Result<ModelDefinition<equation::ODE>> {
    let assay_error = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let residual_error = ResidualErrorModels::new().add(0, ResidualErrorModel::combined(0.5, 0.1));
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(AssayErrorModels::new().add(0, assay_error)?)
        .with_residual_error_models(residual_error);

    ModelDefinition::builder(simple_focei_equation())
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ke", 0.1, 1.0))
                .add(ParameterSpec::bounded("v", 1.0, 20.0)),
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
    saem_reference::assert_close(
        summary.objective_function,
        -425.60904902364695,
        1e-6,
        "npag.objf",
    );
    assert!(summary.converged);
    assert_eq!(summary.iterations, 288);
    assert_eq!(result.get_theta().nspp(), 46);
    saem_reference::assert_close(
        population.parameters[0].mean,
        0.187047284678325,
        1e-6,
        "npag.ke.mean",
    );
    saem_reference::assert_close(
        population.parameters[1].mean,
        107.94241284196241,
        1e-6,
        "npag.v.mean",
    );
    Ok(())
}

#[test]
fn test_acceptance_baseline_saem_bimodal_ke() -> Result<()> {
    let result = bimodal_saem_problem()?.run()?;
    let summary = result.summary();
    let result = result
        .as_parametric()
        .expect("SAEM acceptance baseline should yield a parametric result");

    let mu_psi: Vec<f64> = (0..result.population().npar())
        .map(|index| result.population().mu()[index])
        .collect();
    let omega_diag: Vec<f64> = (0..result.population().npar())
        .map(|index| result.population().omega()[(index, index)])
        .collect();

    // This is the canonical rewrite-blocking parametric baseline for the bimodal_ke example.
    saem_reference::assert_close(
        summary.objective_function,
        -144.18431437030802,
        1e-6,
        "saem.objf",
    );
    assert!(!summary.converged);
    assert_eq!(summary.iterations, 400);
    saem_reference::assert_vec_close(
        &mu_psi,
        &[0.18709059357497426, 105.26324936442889],
        1e-6,
        "saem.mu_psi",
    );
    saem_reference::assert_vec_close(
        &omega_diag,
        &[0.026795214431165025, 489.84880731024896],
        1e-6,
        "saem.omega_diag",
    );
    saem_reference::assert_close(result.sigma().as_vec()[0], 0.102540, 1e-4, "saem.sigma");
    Ok(())
}

#[test]
fn test_acceptance_baseline_focei_onecomp() -> Result<()> {
    let result = EstimationProblem::builder(canonical_focei_model()?, canonical_focei_data())
        .method(EstimationMethod::Parametric(ParametricMethod::Focei(
            FoceiOptions,
        )))
        .output(OutputPlan::disabled())
        .runtime(RuntimeOptions {
            cycles: 3,
            progress: false,
            ..RuntimeOptions::default()
        })
        .run()?;
    let result = result
        .as_parametric()
        .expect("FOCEI acceptance baseline should yield a parametric result");

    let mu: Vec<f64> = (0..result.population().npar())
        .map(|index| result.population().mu()[index])
        .collect();
    let omega_diag: Vec<f64> = (0..result.population().npar())
        .map(|index| result.population().omega()[(index, index)])
        .collect();

    // FOCEI is deterministic on this simple canonical path, so the baseline is exact.
    saem_reference::assert_close(result.objf(), 73.802216624458, 1e-9, "focei.objf");
    saem_reference::assert_vec_close(&mu, &[1.0, 10.5], 1e-12, "focei.mu");
    saem_reference::assert_vec_close(&omega_diag, &[1e-8, 1e-8], 1e-12, "focei.omega_diag");
    assert!(result.sigma().as_vec().is_empty());
    Ok(())
}
