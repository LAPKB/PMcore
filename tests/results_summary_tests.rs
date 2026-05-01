use anyhow::Result;
use pharmsol::{AssayErrorModel, ErrorPoly};
use pmcore::prelude::*;

fn simple_equation() -> equation::ODE {
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
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("results_summary")
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["0"])
            .route(equation::Route::bolus("0").to_state("central")),
    )
    .expect("metadata attachment should validate")
}

fn simple_data() -> Data {
    let subject = Subject::builder("1")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 10.0, 0)
        .observation(2.0, 8.0, 0)
        .build();

    Data::new(vec![subject])
}

#[test]
fn test_nonparametric_fit_result_summary_surface() -> Result<()> {
    let assay_error = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let result = EstimationProblem::builder(simple_equation(), simple_data())
        .parameter(Parameter::bounded("ke", 0.1, 1.0))?
        .parameter(Parameter::bounded("v", 1.0, 20.0))?
        .method(Npag::new())
        .error("0", assay_error)?
        .cycles(1)
        .progress(false)
        .fit()?;

    let summary = result.summary();

    assert_eq!(summary.parameter_count, 2);
    assert_eq!(summary.subject_count, 1);
    assert_eq!(summary.observation_count, 2);
    assert_eq!(result.population_summary().parameters.len(), 2);
    assert_eq!(result.individual_summaries().len(), 1);
    let diagnostics = result.diagnostics();
    assert_eq!(
        diagnostics.estimator_metadata.get("algorithm"),
        Some(&"NPAG".to_string())
    );
    assert_eq!(
        diagnostics.estimator_metadata.get("outputs_requested"),
        Some(&"false".to_string())
    );
    assert!(!diagnostics.convergence_notes.is_empty());
    assert!(diagnostics.deferred_features.is_empty());
    let predictions = result.predictions();
    assert!(!predictions.available);
    assert!(result.artifacts().files.is_empty());
    assert!(result.artifacts().expected_files.is_empty());
    Ok(())
}
