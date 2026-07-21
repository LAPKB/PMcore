use pharmsol::equation::{metadata, AnalyticalKernel, ModelKind, Route};
use pharmsol::prelude::models::one_compartment;
use pharmsol::{fa, fetch_params, lag, Analytical, Equation, Parameters, SubjectBuilderExt, ODE};
use pmcore::{
    AssayErrorModel, AssayErrorModels, AssayLikelihoodError, ErrorModelError, ErrorPoly,
    NormalDistributionError,
};

/// Scientific check that analytical and ODE predictions score to matching
/// likelihoods.
#[test]
fn likelihood_calculation_matches_analytical() {
    let subject = pharmsol::Subject::builder("likelihood")
        .bolus(0.0, 100.0, "iv_bolus")
        .observation(1.0, 1.8, "cp")
        .observation(2.0, 1.6, "cp")
        .observation(4.0, 1.3, "cp")
        .observation(8.0, 0.8, "cp")
        .build();

    let analytical = Analytical::new(
        one_compartment,
        |_p, _t, _cov| {},
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
    )
    .with_nstates(1)
    .with_nout(1)
    .with_ndrugs(1)
    .with_metadata(
        metadata::new("likelihood_calculation")
            .kind(ModelKind::Analytical)
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["cp"])
            .routes([
                Route::bolus("iv_bolus").to_state("central"),
                Route::infusion("iv").to_state("central"),
            ])
            .analytical_kernel(AnalyticalKernel::OneCompartment),
    )
    .unwrap();

    let ode = ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
    )
    .with_nstates(1)
    .with_nout(1)
    .with_ndrugs(1)
    .with_metadata(
        metadata::new("likelihood_calculation")
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["cp"])
            .routes([
                Route::bolus("iv_bolus")
                    .to_state("central")
                    .expect_explicit_input(),
                Route::infusion("iv")
                    .to_state("central")
                    .expect_explicit_input(),
            ]),
    )
    .unwrap();

    let error_models = AssayErrorModels::from(vec![AssayErrorModel::additive(
        ErrorPoly::new(0.0, 0.1, 0.0, 0.0),
        0.0,
    )]);
    let analytical_params = Parameters::with_model(&analytical, [("ke", 0.1), ("v", 50.0)])
        .expect("analytical parameters should validate");
    let ode_params = Parameters::with_model(&ode, [("ke", 0.1), ("v", 50.0)])
        .expect("ODE parameters should validate");

    let analytical_predictions = analytical
        .estimate_predictions(&subject, &analytical_params)
        .expect("analytical predictions");
    let ode_predictions = ode
        .estimate_predictions(&subject, &ode_params)
        .expect("ODE predictions");

    let dense_log_likelihood = error_models
        .log_likelihood(&analytical_predictions)
        .expect("PMcore dense analytical likelihood");
    let named_error_models = AssayErrorModels::new()
        .add(
            "cp",
            AssayErrorModel::additive(ErrorPoly::new(0.0, 0.1, 0.0, 0.0), 0.0),
        )
        .expect("named cp assay model");
    let canonical_outputs = analytical
        .metadata()
        .expect("analytical model metadata")
        .outputs()
        .iter()
        .map(|output| output.name());
    let bound_log_likelihood = named_error_models
        .bind_outputs(canonical_outputs.clone())
        .expect("bind canonical analytical outputs")
        .log_likelihood(&analytical_predictions)
        .expect("score bound named assay model");
    let convenience_log_likelihood = named_error_models
        .log_likelihood_for_outputs(&analytical_predictions, canonical_outputs)
        .expect("bind and score named assay model");
    assert_eq!(bound_log_likelihood, dense_log_likelihood);
    assert_eq!(convenience_log_likelihood, dense_log_likelihood);

    let unbound_multi_output = named_error_models
        .clone()
        .add(
            "effect",
            AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 0.0),
        )
        .expect("second named assay model");
    assert!(matches!(
        unbound_multi_output.log_likelihood(&analytical_predictions),
        Err(AssayLikelihoodError::ErrorModel(
            ErrorModelError::UnboundOutputModels { outputs }
        )) if outputs == ["cp", "effect"]
    ));

    let ll_analytical = dense_log_likelihood.exp();
    let ll_ode = error_models
        .log_likelihood(&ode_predictions)
        .expect("PMcore ODE likelihood")
        .exp();

    let ll_diff = (ll_analytical - ll_ode).abs();
    let ll_rel_diff = ll_diff / ll_analytical.abs().max(1e-10);

    assert!(
        ll_rel_diff < 0.01, // Within 1%
        "Likelihoods should match: analytical={:.6}, ode={:.6}, rel_diff={:.2e}",
        ll_analytical,
        ll_ode,
        ll_rel_diff
    );

    let zero_sigma_models = AssayErrorModels::from(vec![AssayErrorModel::additive(
        ErrorPoly::new(0.0, 0.0, 0.0, 0.0),
        0.0,
    )]);
    assert!(matches!(
        zero_sigma_models.log_likelihood(&analytical_predictions),
        Err(AssayLikelihoodError::Distribution(
            NormalDistributionError::InvalidSigma(sigma)
        )) if sigma == 0.0
    ));

    let impossible_subject = pharmsol::Subject::builder("impossible_likelihood")
        .observation(1.0, 1e308, "cp")
        .build();
    let impossible_predictions = analytical
        .estimate_predictions(&impossible_subject, &analytical_params)
        .expect("analytical impossible-score predictions");
    let constant_sigma_models = AssayErrorModels::from(vec![AssayErrorModel::additive(
        ErrorPoly::new(1.0, 0.0, 0.0, 0.0),
        0.0,
    )]);
    assert!(matches!(
        constant_sigma_models.log_likelihood(&impossible_predictions),
        Err(AssayLikelihoodError::Impossible)
    ));
}
