use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use pharmsol::prelude::*;
use pmcore::algorithms::StopReason;
use pmcore::prelude::*;
use pmcore::results::{InformationCoordinateKind, InformationStatus};

fn output_problem() -> EstimationProblem<pharmsol::equation::Analytical, Parametric> {
    let equation = analytical! {
        name: "saem_output_fixture",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [infusion(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| { y[cp] = x[central] / v; },
    };
    let data = Data::new(vec![
        Subject::builder("s1")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 4.0, "cp")
            .observation(2.0, 3.5, "cp")
            .build(),
        Subject::builder("s2")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 4.5, "cp")
            .observation(2.0, 3.8, "cp")
            .build(),
    ]);
    EstimationProblem::parametric(equation, data)
        .parameter(Parameter::log("ke").with_initial(0.3))
        .parameter(
            Parameter::log("v")
                .with_initial(20.0)
                .fixed()
                .without_random_effect(),
        )
        .omega(Omega::diagonal([("ke", 0.09)]))
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.25)).fixed(),
        )
        .build()
        .expect("output fixture should build")
}

fn fit(compute_map: bool) -> ParametricResult<pharmsol::equation::Analytical> {
    output_problem()
        .fit_with(
            SaemConfig::new()
                .seed(19)
                .n_chains(1)
                .mcmc_iterations(1)
                .burn_in(0)
                .k1_iterations(1)
                .k2_iterations(0)
                .compute_map(compute_map),
        )
        .expect("output fixture should fit")
}

fn fit_iov() -> ParametricResult<pharmsol::equation::Analytical> {
    let equation = analytical! {
        name: "saem_output_iov_fixture",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [infusion(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| { y[cp] = x[central] / v; },
    };
    let data = Data::new(vec![
        Subject::builder("s1")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 4.0, "cp")
            .reset()
            .infusion(12.0, 100.0, "iv", 0.5)
            .observation(13.0, 4.2, "cp")
            .build(),
        Subject::builder("s2")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 4.4, "cp")
            .reset()
            .infusion(12.0, 100.0, "iv", 0.5)
            .observation(13.0, 4.1, "cp")
            .build(),
    ]);
    EstimationProblem::parametric(equation, data)
        .parameter(Parameter::log("ke").with_initial(0.3).fixed())
        .parameter(
            Parameter::log("v")
                .with_initial(20.0)
                .fixed()
                .without_random_effect(),
        )
        .omega(Omega::new().fixed_variance("ke", 0.0225))
        .iov(Iov::new().fixed_variance("ke", 0.04))
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.25)).fixed(),
        )
        .build()
        .expect("IOV output fixture should build")
        .fit_with(
            SaemConfig::new()
                .seed(23)
                .n_chains(1)
                .mcmc_iterations(1)
                .burn_in(0)
                .k1_iterations(1)
                .k2_iterations(0)
                .compute_map(true),
        )
        .expect("IOV output fixture should fit")
}

fn markov_output_equation() -> pharmsol::equation::Analytical {
    analytical! {
        name: "saem_markov_output_fixture",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [infusion(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| { y[cp] = x[central] / v; },
    }
}

fn markov_output_data() -> Data {
    Data::new(vec![
        Subject::builder("s1")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 4.8, "cp")
            .observation(3.0, 3.2, "cp")
            .reset()
            .infusion(12.0, 100.0, "iv", 0.5)
            .observation(13.0, 4.1, "cp")
            .observation(15.0, 2.5, "cp")
            .build(),
        Subject::builder("s2")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 5.3, "cp")
            .observation(3.0, 3.8, "cp")
            .reset()
            .infusion(12.0, 100.0, "iv", 0.5)
            .observation(13.0, 4.6, "cp")
            .observation(15.0, 3.0, "cp")
            .build(),
    ])
}

fn markov_output_problem() -> EstimationProblem<pharmsol::equation::Analytical, Parametric> {
    EstimationProblem::parametric(markov_output_equation(), markov_output_data())
        .parameter(Parameter::log("ke").with_initial(0.25).fixed())
        .parameter(
            Parameter::log("v")
                .with_initial(20.0)
                .fixed()
                .without_random_effect(),
        )
        .omega(Omega::new().fixed_variance("ke", 0.09))
        .iov(Iov::new().fixed_variance("ke", 0.04))
        .error_model("cp", ResidualErrorModel::constant(0.35))
        .build()
        .expect("Markov output fixture should build")
}

fn averaged_iov_problem() -> EstimationProblem<pharmsol::equation::Analytical, Parametric> {
    let equation = analytical! {
        name: "saem_averaged_output_iov_fixture",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [infusion(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| { y[cp] = x[central] / v; },
    };
    let data = Data::new(vec![
        Subject::builder("s1")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 4.8, "cp")
            .observation(3.0, 3.2, "cp")
            .reset()
            .infusion(12.0, 100.0, "iv", 0.5)
            .observation(13.0, 4.1, "cp")
            .observation(15.0, 2.5, "cp")
            .build(),
        Subject::builder("s2")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 5.3, "cp")
            .observation(3.0, 3.8, "cp")
            .reset()
            .infusion(12.0, 100.0, "iv", 0.5)
            .observation(13.0, 4.6, "cp")
            .observation(15.0, 3.0, "cp")
            .build(),
    ]);
    EstimationProblem::parametric(equation, data)
        .parameter(Parameter::log("ke").with_initial(0.25))
        .parameter(
            Parameter::log("v")
                .with_initial(20.0)
                .fixed()
                .without_random_effect(),
        )
        .omega(Omega::diagonal([("ke", 0.09)]))
        .iov(Iov::new().variance("ke", 0.04))
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.35)),
        )
        .build()
        .expect("averaged IOV output fixture should build")
}

fn averaged_iov_config(policy: SaemEstimatorPolicy) -> SaemConfig {
    SaemConfig::new()
        .seed(71_105)
        .n_chains(2)
        .mcmc_iterations(2)
        .burn_in(0)
        .k1_iterations(1)
        .k2_iterations(4)
        .compute_map(true)
        .estimator_policy(policy)
}

fn assert_float_slice_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual
        .iter()
        .zip(expected)
        .all(|(actual, expected)| (actual - expected).abs() < 1e-12));
}

fn assert_json_close(actual: &serde_json::Value, expected: &serde_json::Value) {
    match (actual, expected) {
        (serde_json::Value::Number(actual), serde_json::Value::Number(expected)) => {
            let actual = actual.as_f64().expect("numeric JSON value");
            let expected = expected.as_f64().expect("numeric JSON value");
            assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
        }
        (serde_json::Value::Array(actual), serde_json::Value::Array(expected)) => {
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert_json_close(actual, expected);
            }
        }
        (serde_json::Value::Object(actual), serde_json::Value::Object(expected)) => {
            assert_eq!(actual.len(), expected.len());
            for (key, expected) in expected {
                assert_json_close(&actual[key], expected);
            }
        }
        _ => assert_eq!(actual, expected),
    }
}

fn assert_errorpoly_close(
    actual: Option<pharmsol::ErrorPoly>,
    expected: Option<pharmsol::ErrorPoly>,
) {
    match (actual, expected) {
        (Some(actual), Some(expected)) => {
            let actual = actual.coefficients();
            let expected = expected.coefficients();
            for (actual, expected) in [actual.0, actual.1, actual.2, actual.3]
                .into_iter()
                .zip([expected.0, expected.1, expected.2, expected.3])
            {
                assert!((actual - expected).abs() < 1e-12);
            }
        }
        (None, None) => {}
        _ => panic!("prediction error-polynomial presence differs"),
    }
}

fn constant_mode_objective(
    result: &ParametricResult<pharmsol::equation::Analytical>,
    mode: &pmcore::results::SubjectConditionalMode,
    population: &[f64],
    omega: f64,
    omega_iov: f64,
    sigma: f64,
) -> f64 {
    let log_normal = |value: f64, variance: f64| {
        -0.5 * ((2.0 * std::f64::consts::PI * variance).ln() + value * value / variance)
    };
    let subjects = result.data().subjects();
    let subject = subjects
        .into_iter()
        .find(|subject| subject.id() == &mode.subject_id)
        .expect("mode subject should exist");
    let mut log_posterior = log_normal(mode.eta[0], omega);
    for (occasion, kappa) in subject.occasions().iter().zip(&mode.kappas) {
        assert_eq!(occasion.index(), kappa.occasion_index);
        log_posterior += log_normal(kappa.values[0], omega_iov);
        let parameters = [
            population[0] * (mode.eta[0] + kappa.values[0]).exp(),
            population[1],
        ];
        let occasion_subject =
            Subject::from_occasions(subject.id().clone(), vec![occasion.clone()]);
        let predictions = result
            .equation()
            .estimate_predictions_dense(&occasion_subject, &parameters)
            .expect("fresh occasion prediction should succeed");
        for prediction in predictions.predictions() {
            if let Some(observation) = prediction.observation() {
                let residual = observation - prediction.prediction();
                log_posterior += log_normal(residual, sigma * sigma);
            }
        }
    }
    -log_posterior
}

fn fit_without_random_effects() -> ParametricResult<pharmsol::equation::Analytical> {
    let equation = analytical! {
        name: "saem_output_no_random_effect_fixture",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [infusion(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| { y[cp] = x[central] / v; },
    };
    let data = Data::new(vec![Subject::builder("s1")
        .infusion(0.0, 100.0, "iv", 0.5)
        .observation(1.0, 4.0, "cp")
        .build()]);
    EstimationProblem::parametric(equation, data)
        .parameter(
            Parameter::log("ke")
                .with_initial(0.3)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::log("v")
                .with_initial(20.0)
                .fixed()
                .without_random_effect(),
        )
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.25)).fixed(),
        )
        .build()
        .expect("no-random-effect output fixture should build")
        .fit_with(
            SaemConfig::new()
                .seed(29)
                .n_chains(1)
                .mcmc_iterations(1)
                .burn_in(0)
                .k1_iterations(1)
                .k2_iterations(0)
                .compute_map(false),
        )
        .expect("no-random-effect output fixture should fit")
}

fn fit_covariate_individual_output(iov: bool) -> ParametricResult<pharmsol::equation::Analytical> {
    let equation = analytical! {
        name: "saem_covariate_individual_output",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [bolus(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| { y[cp] = x[central] / v; },
    };
    let subject = |id: &str, wt: f64, group: f64| {
        let mut builder = Subject::builder(id)
            .covariate("wt", 0.0, wt)
            .covariate("group", 0.0, group)
            .bolus(0.0, 100.0, "iv")
            .observation(1.0, 3.0, "cp");
        if iov {
            builder = builder
                .reset()
                .covariate("wt", 12.0, wt)
                .covariate("group", 12.0, group)
                .bolus(12.0, 100.0, "iv")
                .observation(13.0, 3.0, "cp");
        }
        builder.build()
    };
    let data = Data::new(vec![subject("s1", 0.0, 0.0), subject("s2", 10.0, 1.0)]);
    let builder = EstimationProblem::parametric(equation, data)
        .parameter(
            Parameter::log("ke")
                .with_initial(0.2)
                .fixed()
                .with_random_effect(iov),
        )
        .parameter(
            Parameter::log("v")
                .with_initial(20.0)
                .fixed()
                .without_random_effect(),
        )
        .covariate_effect(
            CovariateEffect::continuous("ke", "wt", 0.0)
                .with_initial(0.01)
                .fixed(),
        )
        .covariate_effect(
            CovariateEffect::categorical("ke", "group", 0.0, 1.0)
                .with_initial(0.2)
                .fixed(),
        )
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.25)).fixed(),
        );
    let problem = if iov {
        builder
            .omega(Omega::new().fixed_variance("ke", 0.04))
            .iov(Iov::new().fixed_variance("ke", 0.09))
            .build()
    } else {
        builder.build()
    }
    .expect("covariate individual-output fixture should build");
    problem
        .fit_with(
            SaemConfig::new()
                .seed(30_031)
                .n_chains(1)
                .mcmc_iterations(1)
                .burn_in(0)
                .k1_iterations(1)
                .k2_iterations(0)
                .compute_map(false),
        )
        .expect("covariate individual-output fixture should fit")
}

fn fit_structured_omega() -> ParametricResult<pharmsol::equation::Analytical> {
    let equation = analytical! {
        name: "saem_output_structured_omega_fixture",
        params: [ke, v, bio],
        states: [central],
        outputs: [cp],
        routes: [infusion(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| { y[cp] = bio * x[central] / v; },
    };
    let data = Data::new(vec![Subject::builder("s1")
        .infusion(0.0, 100.0, "iv", 0.5)
        .observation(1.0, 4.0, "cp")
        .observation(2.0, 3.5, "cp")
        .build()]);
    EstimationProblem::parametric(equation, data)
        .parameter(Parameter::log("ke").with_initial(0.3).fixed())
        .parameter(Parameter::log("v").with_initial(20.0).fixed())
        .parameter(Parameter::log("bio").with_initial(1.0).fixed())
        .omega(
            Omega::new()
                .variance("ke", 0.09)
                .fixed_variance("v", 1.0)
                .variance("bio", 0.04)
                .fixed_covariance("ke", "v", 0.01)
                .covariance("v", "bio", 0.02),
        )
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.25)).fixed(),
        )
        .build()
        .expect("structured Omega fixture should build")
        .fit_with(
            SaemConfig::new()
                .seed(31)
                .n_chains(1)
                .mcmc_iterations(1)
                .burn_in(0)
                .k1_iterations(1)
                .k2_iterations(0)
                .compute_map(false),
        )
        .expect("structured Omega fixture should fit")
}

fn fit_residual_output(
    residual: ParametricErrorModel,
    seed: u64,
) -> ParametricResult<pharmsol::equation::Analytical> {
    let equation = analytical! {
        name: "saem_output_residual_fixture",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [infusion(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| { y[cp] = x[central] / v; },
    };
    let data = Data::new(vec![Subject::builder("s1")
        .infusion(0.0, 100.0, "iv", 0.5)
        .observation(1.0, 4.0, "cp")
        .observation(2.0, 3.5, "cp")
        .observation(4.0, 2.5, "cp")
        .build()]);
    EstimationProblem::parametric(equation, data)
        .parameter(
            Parameter::log("ke")
                .with_initial(0.3)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::log("v")
                .with_initial(20.0)
                .fixed()
                .without_random_effect(),
        )
        .error_model("cp", residual)
        .build()
        .expect("combined output fixture should build")
        .fit_with(
            SaemConfig::new()
                .seed(seed)
                .n_chains(1)
                .mcmc_iterations(1)
                .burn_in(0)
                .k1_iterations(1)
                .k2_iterations(0)
                .compute_map(false),
        )
        .expect("residual output fixture should fit")
}

fn fit_combined_residual(fixed_additive: bool) -> ParametricResult<pharmsol::equation::Analytical> {
    let residual = ParametricErrorModel::new(ResidualErrorModel::combined(0.25, 0.05));
    let residual = if fixed_additive {
        residual.fixed_combined_additive()
    } else {
        residual.fixed_combined_proportional()
    };
    fit_residual_output(residual, 37)
}

fn temp_output_dir(label: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should follow epoch")
        .as_nanos();
    std::path::Path::new("target")
        .join("saem-output-tests")
        .join(format!("{label}-{nonce}"))
}

#[test]
fn tables_preserve_order_masks_and_optional_conditionals() {
    let result = fit(false);
    let tables = result.tables(0.0, 0.0).expect("tables should build");
    let summary = result.population_summary();
    assert_eq!(summary.parameters.len(), 2);
    for (parameter, estimate) in summary
        .parameters
        .iter()
        .zip(result.population_parameters())
    {
        assert_eq!(parameter.estimate, *estimate);
        assert_eq!(parameter.mean, None);
        assert_eq!(parameter.median, None);
        assert_eq!(parameter.sd, None);
        assert_eq!(parameter.cv_percent, None);
    }

    assert_eq!(tables.population.len(), 2);
    assert_eq!(tables.population[0].name, "ke");
    assert!(tables.population[0].estimated);
    assert!(tables.population[0].iiv);
    assert_eq!(tables.population[1].name, "v");
    assert!(!tables.population[1].estimated);
    assert!(!tables.population[1].iiv);
    assert_eq!(tables.omega.len(), 1);
    assert!(tables.omega[0].structural);
    assert!(tables.omega[0].estimated);
    assert!(tables.omega_iov.is_none());
    assert_eq!(tables.residual_error.len(), 1);
    assert_eq!(tables.residual_error[0].component, "sigma");
    assert!(!tables.residual_error[0].estimated);
    assert_eq!(tables.individual_effects.len(), 2);
    assert_eq!(tables.individual_effects[0].subject, "s1");
    assert_eq!(tables.individual_effects[0].parameter, "ke");
    assert_eq!(tables.individual_effects[0].source, "chain_mean");
    assert_eq!(tables.individual_parameters.len(), 4);
    assert_eq!(tables.individual_parameters[0].subject, "s1");
    assert_eq!(tables.individual_parameters[0].occasion, None);
    assert_eq!(tables.individual_parameters[0].parameter, "ke");
    assert_eq!(tables.individual_parameters[1].parameter, "v");
    assert_eq!(tables.individual_parameters[2].subject, "s2");
    assert_eq!(tables.iterations.len(), 1);
    assert_eq!(tables.iterations[0].cycle, 1);
    assert!(tables.statistics.iter().any(|row| row.kind == "theta"));
    assert!(tables.statistics.iter().any(|row| row.kind == "omega"));
    assert!(tables.statistics.iter().any(|row| row.kind == "residual"));
    assert!(!tables.predictions.is_empty());
    assert!(tables
        .predictions
        .iter()
        .all(|row| row.conditional_prediction.is_none() && row.conditional_source.is_none()));
}

#[test]
fn iov_tables_include_lower_triangle_and_ordered_kappas() {
    let result = fit_iov();
    let tables = result.tables(0.0, 0.0).expect("IOV tables should build");
    let omega_iov = tables.omega_iov.expect("IOV table should be present");
    assert_eq!(omega_iov.len(), 1);
    assert_eq!(omega_iov[0].row, "ke");
    assert_eq!(omega_iov[0].column, "ke");
    assert!(omega_iov[0].structural);
    assert!(!omega_iov[0].estimated);
    let kappas: Vec<_> = tables
        .individual_effects
        .iter()
        .filter(|row| row.effect_kind == "kappa" && row.source == "chain_mean")
        .collect();
    assert_eq!(kappas.len(), 4);
    assert!(kappas.iter().all(|row| row.occasion.is_some()));

    for source in ["chain_mean", "conditional_mode"] {
        let parameters: Vec<_> = tables
            .individual_parameters
            .iter()
            .filter(|row| row.source == source)
            .collect();
        assert_eq!(parameters.len(), 8);
        let keys: Vec<_> = parameters
            .iter()
            .map(|row| (row.subject.as_str(), row.occasion, row.parameter.as_str()))
            .collect();
        assert_eq!(
            keys,
            vec![
                ("s1", Some(0), "ke"),
                ("s1", Some(0), "v"),
                ("s1", Some(1), "ke"),
                ("s1", Some(1), "v"),
                ("s2", Some(0), "ke"),
                ("s2", Some(0), "v"),
                ("s2", Some(1), "ke"),
                ("s2", Some(1), "v"),
            ]
        );
        assert_ne!(parameters[0].value, parameters[2].value);
        assert!((parameters[1].value - 20.0).abs() < 1e-12);
        assert!((parameters[3].value - 20.0).abs() < 1e-12);
        for row in parameters.iter().filter(|row| row.parameter == "ke") {
            let occasion = row.occasion.expect("IOV row should name an occasion");
            let (eta, kappa) = if source == "chain_mean" {
                let eta = result
                    .eta_chain_mean(&row.subject)
                    .expect("chain eta should exist")
                    .values[0];
                let kappa = result
                    .kappa_chain_mean(&row.subject, occasion)
                    .expect("chain kappa should exist")
                    .values[0];
                (eta, kappa)
            } else {
                let mode = result
                    .conditional_mode(&row.subject)
                    .expect("conditional mode should exist");
                let kappa = mode
                    .kappas
                    .iter()
                    .find(|value| value.occasion_index == occasion)
                    .expect("mode kappa should exist")
                    .values[0];
                (mode.eta[0], kappa)
            };
            let expected = result.population_parameters()[0] * (eta + kappa).exp();
            assert!((row.value - expected).abs() < 1e-12);
        }
    }

    let population = result
        .population_predictions(0.25, 0.0)
        .expect("dense population predictions should build");
    let conditional = result
        .conditional_predictions(0.25, 0.0)
        .expect("dense conditional predictions should build");
    for (population_subject, conditional_subject) in population.iter().zip(&conditional) {
        assert_eq!(
            population_subject.predictions().len(),
            conditional_subject.predictions().len()
        );
        for (population_point, conditional_point) in population_subject
            .predictions()
            .iter()
            .zip(conditional_subject.predictions())
        {
            assert_eq!(population_point.time(), conditional_point.time());
            assert_eq!(population_point.outeq(), conditional_point.outeq());
            assert_eq!(population_point.occasion(), conditional_point.occasion());
            assert_eq!(
                population_point.observation(),
                conditional_point.observation()
            );
            assert_eq!(population_point.censoring(), conditional_point.censoring());
        }
    }

    let dense = result
        .tables(0.25, 0.0)
        .expect("dense IOV tables should build");
    assert!(dense.predictions.len() > 4);
    assert!(dense.predictions.iter().all(|row| {
        row.conditional_prediction.is_some()
            && row.conditional_source.as_deref() == Some("conditional_mode")
            && row.block <= 1
    }));
}

#[test]
fn averaged_iov_result_rebuilds_all_deterministic_outputs_from_canonical_state() {
    let averaged = averaged_iov_problem()
        .fit_with(averaged_iov_config(SaemEstimatorPolicy::AveragedIterates {
            alpha: 0.75,
        }))
        .expect("averaged IOV output fixture should fit");
    assert_eq!(averaged.termination_reason(), Some(&StopReason::MaxCycles));
    assert!(!averaged.converged());
    assert!(averaged.estimator_metadata().average_applied);
    assert_eq!(averaged.estimator_metadata().averaging_start_cycle, Some(2));
    assert_eq!(averaged.estimator_metadata().averaged_iterations, 4);
    assert_eq!(averaged.cycle_diagnostics().len(), 5);
    let terminal_cycle = averaged
        .cycle_diagnostics()
        .last()
        .expect("terminal cycle diagnostic");
    assert_eq!(terminal_cycle.iteration, 5);
    assert_eq!(terminal_cycle.phase, pmcore::results::SaemPhase::Smoothing);

    let installed_terminal_population_gap = averaged
        .population_parameters()
        .iter()
        .zip(&terminal_cycle.population_parameters)
        .map(|(average, terminal)| (average - terminal).abs())
        .fold(0.0_f64, f64::max);
    assert!(installed_terminal_population_gap > 1e-4);
    assert!((averaged.omega()[[0, 0]] - terminal_cycle.omega[[0, 0]]).abs() > 1e-6);
    assert!(
        (averaged.omega_iov().unwrap()[[0, 0]]
            - terminal_cycle.omega_iov.as_ref().unwrap()[[0, 0]])
        .abs()
            > 1e-6
    );
    let averaged_sigma = averaged.residual_error_estimates()[0].model.sigma(1.0);
    let terminal_sigma = terminal_cycle.residual_error_estimates[0].model.sigma(1.0);
    assert!((averaged_sigma - terminal_sigma).abs() > 1e-5);

    assert_eq!(averaged.conditional_modes().len(), 2);
    for mode in averaged.conditional_modes() {
        let fresh_objective = constant_mode_objective(
            &averaged,
            mode,
            averaged.population_parameters(),
            averaged.omega()[[0, 0]],
            averaged.omega_iov().unwrap()[[0, 0]],
            averaged_sigma,
        );
        assert!((mode.objective - fresh_objective).abs() < 1e-9);
        let terminal_state_objective = constant_mode_objective(
            &averaged,
            mode,
            &terminal_cycle.population_parameters,
            terminal_cycle.omega[[0, 0]],
            terminal_cycle.omega_iov.as_ref().unwrap()[[0, 0]],
            terminal_sigma,
        );
        assert!((mode.objective - terminal_state_objective).abs() > 1e-4);

        assert!(
            (mode.parameters[0] - averaged.population_parameters()[0] * mode.eta[0].exp()).abs()
                < 1e-12
        );
        assert!((mode.parameters[1] - averaged.population_parameters()[1]).abs() < 1e-12);
    }

    let tables = averaged.tables(0.25, 0.0).expect("averaged tables");
    let mode_rows: Vec<_> = tables
        .individual_parameters
        .iter()
        .filter(|row| row.source == "conditional_mode")
        .collect();
    assert_eq!(mode_rows.len(), 8);
    for row in mode_rows {
        let mode = averaged.conditional_mode(&row.subject).unwrap();
        let occasion = row.occasion.expect("IOV mode row should name an occasion");
        let kappa = mode
            .kappas
            .iter()
            .find(|kappa| kappa.occasion_index == occasion)
            .unwrap();
        let expected = match row.parameter.as_str() {
            "ke" => averaged.population_parameters()[0] * (mode.eta[0] + kappa.values[0]).exp(),
            "v" => averaged.population_parameters()[1],
            parameter => panic!("unexpected parameter row {parameter}"),
        };
        assert!((row.value - expected).abs() < 1e-12);
    }

    let expanded = averaged.data().clone().expand(0.25, 0.0);
    let population = averaged.population_predictions(0.25, 0.0).unwrap();
    let mut differs_from_terminal_prediction = false;
    for (subject, actual) in expanded.subjects().iter().zip(&population) {
        let expected = averaged
            .equation()
            .estimate_predictions_dense(subject, averaged.population_parameters())
            .unwrap();
        let terminal_expected = averaged
            .equation()
            .estimate_predictions_dense(subject, &terminal_cycle.population_parameters)
            .unwrap();
        assert_eq!(actual.predictions().len(), expected.predictions().len());
        for ((actual, expected), terminal_expected) in actual
            .predictions()
            .iter()
            .zip(expected.predictions())
            .zip(terminal_expected.predictions())
        {
            assert_eq!(actual.time(), expected.time());
            assert!((actual.prediction() - expected.prediction()).abs() < 1e-12);
            assert_eq!(actual.observation(), expected.observation());
            assert_eq!(actual.outeq(), expected.outeq());
            assert_errorpoly_close(actual.errorpoly(), expected.errorpoly());
            assert_float_slice_close(actual.state(), expected.state());
            assert_eq!(actual.occasion(), expected.occasion());
            assert_eq!(actual.censoring(), expected.censoring());
            differs_from_terminal_prediction |=
                (actual.prediction() - terminal_expected.prediction()).abs() > 1e-5;
        }
    }
    assert!(differs_from_terminal_prediction);

    let conditional = averaged.conditional_predictions(0.25, 0.0).unwrap();
    for (subject, actual) in expanded.subjects().iter().zip(&conditional) {
        let mode = averaged.conditional_mode(subject.id()).unwrap();
        let mut expected = Vec::new();
        for (occasion, kappa) in subject.occasions().iter().zip(&mode.kappas) {
            assert_eq!(occasion.index(), kappa.occasion_index);
            let parameters = [
                averaged.population_parameters()[0] * (mode.eta[0] + kappa.values[0]).exp(),
                averaged.population_parameters()[1],
            ];
            let occasion_subject =
                Subject::from_occasions(subject.id().clone(), vec![occasion.clone()]);
            expected.extend(
                averaged
                    .equation()
                    .estimate_predictions_dense(&occasion_subject, &parameters)
                    .unwrap()
                    .predictions()
                    .iter()
                    .cloned(),
            );
        }
        assert_eq!(actual.predictions().len(), expected.len());
        for (actual, expected) in actual.predictions().iter().zip(&expected) {
            assert_eq!(actual.time(), expected.time());
            assert!((actual.prediction() - expected.prediction()).abs() < 1e-12);
            assert_eq!(actual.observation(), expected.observation());
            assert_eq!(actual.outeq(), expected.outeq());
            assert_errorpoly_close(actual.errorpoly(), expected.errorpoly());
            assert_float_slice_close(actual.state(), expected.state());
            assert_eq!(actual.occasion(), expected.occasion());
            assert_eq!(actual.censoring(), expected.censoring());
        }
    }
}

#[test]
fn covariance_rows_use_named_lower_triangle_order_and_masks() {
    let tables = fit_structured_omega()
        .tables(0.0, 0.0)
        .expect("structured tables should build");
    let actual: Vec<_> = tables
        .omega
        .iter()
        .map(|row| {
            (
                row.row.as_str(),
                row.column.as_str(),
                row.structural,
                row.estimated,
            )
        })
        .collect();
    assert_eq!(
        actual,
        vec![
            ("ke", "ke", true, true),
            ("v", "ke", true, false),
            ("v", "v", true, false),
            ("bio", "ke", false, false),
            ("bio", "v", true, true),
            ("bio", "bio", true, true),
        ]
    );
}

#[test]
fn standalone_proportional_coordinate_joins_persisted_residual_row() {
    let result = fit_residual_output(
        ParametricErrorModel::new(ResidualErrorModel::proportional(0.05)),
        39,
    );
    let row = result
        .tables(0.0, 0.0)
        .unwrap()
        .residual_error
        .into_iter()
        .next()
        .unwrap();
    assert_eq!(row.component, "proportional");
    assert!(result
        .information_diagnostics()
        .coordinates
        .iter()
        .any(|coordinate| matches!(
            &coordinate.kind,
            InformationCoordinateKind::Residual {
                output_index: 0,
                component,
            } if component == "proportional"
        ) && coordinate.name == "residual:cp:proportional"));
}

#[test]
fn combined_residual_rows_preserve_independent_component_masks() {
    for (fixed_additive, expected) in [
        (true, [("additive", false), ("proportional", true)]),
        (false, [("additive", true), ("proportional", false)]),
    ] {
        let tables = fit_combined_residual(fixed_additive)
            .tables(0.0, 0.0)
            .expect("combined tables should build");
        let actual: Vec<_> = tables
            .residual_error
            .iter()
            .map(|row| (row.component.as_str(), row.estimated))
            .collect();
        assert_eq!(actual, expected);

        let result = fit_combined_residual(fixed_additive);
        let coordinate_components: Vec<_> = result
            .information_diagnostics()
            .coordinates
            .iter()
            .filter_map(|coordinate| match &coordinate.kind {
                InformationCoordinateKind::Residual {
                    output_index,
                    component,
                } => Some((*output_index, component.as_str())),
                _ => None,
            })
            .collect();
        for row in result.tables(0.0, 0.0).unwrap().residual_error {
            if row.estimated {
                assert!(coordinate_components.contains(&(row.output_index, row.component.as_str())));
            }
        }
    }
}

#[test]
fn nondifferentiable_information_floor_does_not_fail_a_valid_fit() {
    let equation = analytical! {
        name: "saem_information_floor_fixture",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [infusion(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| { y[cp] = x[central] / v; },
    };
    let data = Data::new(vec![Subject::builder("s1")
        .infusion(0.0, 100.0, "iv", 0.5)
        .observation(1.0, 4.0, "cp")
        .build()]);
    let result = EstimationProblem::parametric(equation, data)
        .parameter(
            Parameter::log("ke")
                .with_initial(0.3)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::log("v")
                .with_initial(20.0)
                .fixed()
                .without_random_effect(),
        )
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(f64::EPSILON.sqrt())).fixed(),
        )
        .build()
        .unwrap()
        .fit_with(
            SaemConfig::new()
                .seed(41)
                .n_chains(1)
                .mcmc_iterations(1)
                .burn_in(0)
                .k1_iterations(1)
                .k2_iterations(0)
                .compute_map(false),
        )
        .expect("the fit remains scientifically valid when only information is unavailable");
    assert_eq!(
        result.termination_reason(),
        Some(&pmcore::algorithms::StopReason::MaxCycles)
    );
    assert_eq!(
        result.information_diagnostics().status,
        InformationStatus::Ineligible(
            "constant residual scale is exactly at the nondifferentiable likelihood floor boundary"
                .to_string()
        )
    );
}

#[test]
fn zero_random_effect_predictions_are_available_without_conditional_modes() {
    let result = fit_without_random_effects();
    assert!(result.conditional_modes().is_empty());
    let tables = result.tables(0.25, 0.0).expect("tables should build");
    assert_eq!(tables.individual_parameters.len(), 2);
    assert!(tables
        .individual_parameters
        .iter()
        .all(|row| row.occasion.is_none() && row.source == "chain_mean"));
    assert!(!tables.predictions.is_empty());
    assert!(tables.predictions.iter().all(|row| {
        row.conditional_prediction == Some(row.population_prediction)
            && row.conditional_source.as_deref() == Some("population")
    }));
    let directory = temp_output_dir("zero-random-schema-seven");
    result.write_outputs(&directory, 0.25, 0.0).unwrap();
    let persisted = ParametricResultRecord::read_json(directory.join("result.json")).unwrap();
    assert_eq!(persisted.source_metadata.omega.dimension, 0);
    assert!(persisted.source_metadata.omega.names.is_empty());
    assert!(persisted.source_metadata.omega.values.is_empty());
    assert!(persisted.source_metadata.omega.structural_mask.is_empty());
    assert!(persisted.source_metadata.omega.estimated_mask.is_empty());
    assert!(persisted.source_metadata.covariate_effects.is_empty());
    assert!(persisted.source_metadata.subject_covariates.is_empty());
    assert!(persisted.source_metadata.subject_design.is_empty());
    assert!(persisted
        .source_metadata
        .subject_population_parameters
        .is_empty());
    assert_eq!(persisted.source_metadata.residual_outputs[0].values, [0.25]);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn covariate_subject_means_drive_no_iov_individual_and_prediction_outputs() {
    let result = fit_covariate_individual_output(false);
    assert!(result
        .eta_chain_means()
        .iter()
        .all(|eta| eta.values.is_empty()));
    let tables = result.tables(0.0, 0.0).unwrap();
    let ke = |subject: &str, rows: &[IndividualParameterRow]| {
        rows.iter()
            .find(|row| row.subject == subject && row.parameter == "ke")
            .unwrap()
            .value
    };
    let expected_s1 = 0.2;
    let expected_s2 = 0.2 * 0.3f64.exp();
    assert!((ke("s1", &tables.individual_parameters) - expected_s1).abs() <= 1e-12);
    assert!((ke("s2", &tables.individual_parameters) - expected_s2).abs() <= 1e-12);
    assert_ne!(expected_s1, expected_s2);
    for (subject, expected_ke) in [("s1", expected_s1), ("s2", expected_s2)] {
        let prediction = tables
            .predictions
            .iter()
            .find(|row| row.subject == subject && row.time == 1.0)
            .unwrap();
        let expected_prediction = 100.0 * (-expected_ke).exp() / 20.0;
        assert!((prediction.population_prediction - expected_prediction).abs() <= 1e-12);
    }

    let directory = temp_output_dir("covariate-no-iov-individual");
    result.write_outputs(&directory, 0.0, 0.0).unwrap();
    let csv = fs::read_to_string(directory.join("individual_parameters.csv")).unwrap();
    assert!(csv.contains("s1,,ke,0.2,chain_mean,"));
    assert!(csv.contains("s2,,ke,"));
    let persisted = ParametricResultRecord::read_json(directory.join("result.json")).unwrap();
    assert_eq!(
        persisted.tables.individual_parameters,
        tables.individual_parameters
    );
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn schema_seven_reader_rejects_coordinated_fixed_beta_tampering() {
    let result = fit_covariate_individual_output(false);
    let directory = temp_output_dir("fixed-beta-tampering");
    result.write_outputs(&directory, 0.0, 0.0).unwrap();
    let result_path = directory.join("result.json");
    let mut raw: serde_json::Value =
        serde_json::from_reader(fs::File::open(&result_path).unwrap()).unwrap();
    let changed = raw["source_metadata"]["covariate_effects"][0]["estimate"]
        .as_f64()
        .unwrap()
        + 0.01;
    raw["source_metadata"]["covariate_effects"][0]["estimate"] = serde_json::json!(changed);
    raw["tables"]["covariate_effects"][0]["estimate"] = serde_json::json!(changed);
    for row in raw["tables"]["statistics"].as_array_mut().unwrap() {
        if row["kind"] == "covariate_effect_final" && row["name"] == "beta:ke:wt" {
            row["value"] = serde_json::json!(changed);
        }
    }
    fs::write(&result_path, serde_json::to_vec_pretty(&raw).unwrap()).unwrap();
    assert!(ParametricResultRecord::read_json(&result_path).is_err());
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn covariate_subject_means_drive_iov_occasion_individual_outputs() {
    let result = fit_covariate_individual_output(true);
    let tables = result.tables(0.0, 0.0).unwrap();
    for subject in ["s1", "s2"] {
        let mu = if subject == "s1" {
            0.2
        } else {
            0.2 * 0.3f64.exp()
        };
        let eta = result.eta_chain_mean(subject).unwrap().values[0];
        for occasion in 0..2 {
            let kappa = result.kappa_chain_mean(subject, occasion).unwrap().values[0];
            let row = tables
                .individual_parameters
                .iter()
                .find(|row| {
                    row.subject == subject
                        && row.occasion == Some(occasion)
                        && row.parameter == "ke"
                        && row.source == "chain_mean"
                })
                .unwrap();
            let expected = (mu.ln() + eta + kappa).exp();
            assert!((row.value - expected).abs() <= 1e-12);
            let global_fallback = (0.2f64.ln() + eta + kappa).exp();
            if subject == "s2" {
                assert!((row.value - global_fallback).abs() > 1e-6);
            }
        }
    }
    let directory = temp_output_dir("covariate-iov-individual");
    result.write_outputs(&directory, 0.0, 0.0).unwrap();
    let persisted = ParametricResultRecord::read_json(directory.join("result.json")).unwrap();
    assert_eq!(
        persisted.tables.individual_parameters,
        tables.individual_parameters
    );
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn output_bundle_has_exact_files_headers_and_loadable_json() {
    let result = fit(true);
    let directory = temp_output_dir("bundle");
    result
        .write_outputs(&directory, 0.0, 0.0)
        .expect("output bundle should write");

    let required = [
        "population.csv",
        "omega.csv",
        "residual_error.csv",
        "individual_effects.csv",
        "individual_parameters.csv",
        "iterations.csv",
        "statistics.csv",
        "marginal_likelihood.csv",
        "information_criteria.csv",
        "predictions.csv",
        "covariate_effects.csv",
        "subject_covariates.csv",
        "subject_population_parameters.csv",
        "result.json",
        "manifest.json",
    ];
    let mut entries: Vec<_> = fs::read_dir(&directory)
        .expect("output directory should read")
        .map(|entry| {
            entry
                .expect("output entry should read")
                .file_name()
                .into_string()
                .expect("output filename should be UTF-8")
        })
        .collect();
    entries.sort();
    let mut expected_entries = required.map(str::to_string).to_vec();
    expected_entries.sort();
    assert_eq!(entries, expected_entries);

    let population =
        fs::read_to_string(directory.join("population.csv")).expect("population CSV should read");
    assert_eq!(
        population.lines().next(),
        Some("name,estimate,scale,estimated,iiv,iov")
    );
    let predictions =
        fs::read_to_string(directory.join("predictions.csv")).expect("predictions CSV should read");
    assert_eq!(
        predictions.lines().next(),
        Some("subject,time,output_index,block,observation,censoring,population_prediction,conditional_prediction,conditional_source")
    );

    let statistics =
        fs::read_to_string(directory.join("statistics.csv")).expect("statistics CSV should read");
    assert_eq!(
        statistics.lines().next(),
        Some("cycle,kind,name,row,column,output_index,component,value,status")
    );

    let marginal = fs::read_to_string(directory.join("marginal_likelihood.csv"))
        .expect("marginal likelihood CSV should read");
    assert!(marginal.lines().next().is_some_and(|header| {
        header.starts_with("scope,subject,method,status,samples_per_subject,seed")
    }));

    let individual_parameters = fs::read_to_string(directory.join("individual_parameters.csv"))
        .expect("individual parameter CSV should read");
    assert_eq!(
        individual_parameters.lines().next(),
        Some("subject,occasion,parameter,value,source,mode_converged")
    );

    let manifest: serde_json::Value = serde_json::from_reader(
        fs::File::open(directory.join("manifest.json")).expect("manifest should open"),
    )
    .expect("manifest should parse");
    assert_eq!(
        manifest,
        serde_json::json!({
            "schema_version": 9,
            "fit_family": "parametric",
            "algorithm": "saem",
            "objective_kind": "conditional_n2ll",
            "termination": "MaxCycles",
            "operational_convergence": {
                "config": null,
                "checks": [],
                "final_check_reused": false,
                "used_for_termination": false,
                "final_status": null,
                "worst_rhat": null,
                "min_bulk_ess": null,
                "fixed_width_ratio": null,
                "fixed_width_epsilon": null,
                "implied_minimum_ess": null,
                "newton_displacement": null,
                "newton_displacement_mc_sd": null
            },
            "estimator_metadata": {
                "policy": "TerminalIterate",
                "average_applied": false,
                "averaging_start_cycle": null,
                "averaged_iterations": 0
            },
            "marginal_likelihood": null,
            "information_criteria": result.information_criteria(),
            "files": [
                "population.csv",
                "omega.csv",
                "residual_error.csv",
                "individual_effects.csv",
                "individual_parameters.csv",
                "iterations.csv",
                "statistics.csv",
                "marginal_likelihood.csv",
                "information_criteria.csv",
                "predictions.csv",
                "covariate_effects.csv",
                "subject_covariates.csv",
                "subject_population_parameters.csv",
                "result.json",
                "manifest.json"
            ]
        })
    );

    let expected_tables = result.tables(0.0, 0.0).expect("tables should rebuild");
    let expected_config_seed = result.config().seed;
    let expected_termination = result.termination_reason().cloned();
    drop(result);

    let record = ParametricResultRecord::read_json(directory.join("result.json"))
        .expect("equation-free result should load");
    assert_eq!(record.schema_version, 9);
    assert_eq!(record.fit_family, "parametric");
    assert_eq!(record.algorithm, "saem");
    assert_eq!(record.objective_kind, "conditional_n2ll");
    assert_eq!(record.config.seed, expected_config_seed);
    assert_eq!(record.termination, expected_termination);
    assert_eq!(record.tables, expected_tables);
    let information = &record.information_diagnostics;
    assert!(!information.coordinates.is_empty());
    assert!(information
        .observed_information
        .iter()
        .flatten()
        .all(|value| value.is_finite()));
    for row in 0..information.observed_information.len() {
        for column in 0..row {
            assert!(
                (information.observed_information[row][column]
                    - information.observed_information[column][row])
                    .abs()
                    < 1e-12
            );
        }
    }
    let information_rows: Vec<_> = record
        .tables
        .statistics
        .iter()
        .filter(|row| {
            (row.kind.starts_with("information_") && !row.kind.starts_with("information_criteria"))
                || row.kind == "observed_information"
        })
        .collect();
    assert!(!information_rows.is_empty());
    let expected_status = match &information.status {
        InformationStatus::Available => "available".to_string(),
        InformationStatus::NoFreeCoordinates => "no_free_coordinates".to_string(),
        InformationStatus::NonFinite => "non_finite".to_string(),
        InformationStatus::ObservedInformationNotPositiveDefinite => {
            "observed_information_not_positive_definite".to_string()
        }
        InformationStatus::Unsupported(reason) => format!("unsupported: {reason}"),
        InformationStatus::Ineligible(reason) => format!("ineligible: {reason}"),
    };
    assert!(information_rows
        .iter()
        .all(|row| row.status.as_deref() == Some(expected_status.as_str())));
    let markov_rows: Vec<_> = record
        .tables
        .statistics
        .iter()
        .filter(|row| row.kind.starts_with("markov_"))
        .collect();
    assert!(!markov_rows.is_empty());
    assert!(markov_rows.iter().all(|row| row.status.is_some()));
    assert!(record
        .tables
        .statistics
        .iter()
        .filter(|row| {
            !row.kind.starts_with("information_")
                && row.kind != "observed_information"
                && !row.kind.starts_with("markov_")
                && !row.kind.starts_with("marginal_likelihood")
                && !row.kind.starts_with("population_uncertainty")
                && !row.kind.starts_with("conditional_")
                && !row.kind.contains("shrinkage")
        })
        .all(|row| row.status.is_none()));
    assert!(record
        .tables
        .predictions
        .iter()
        .all(|row| row.conditional_prediction.is_some()
            && row.conditional_source.as_deref() == Some("conditional_mode")));

    fs::remove_dir_all(directory).expect("temporary output directory should be removable");
}

#[test]
fn enabled_schema_six_preserves_markov_config_status_raw_matrices_and_warm_start() {
    use pmcore::algorithms::parametric::{
        LugsailConfig, MarkovSimulationVarianceConfig, SaemEstimatorPolicy,
    };

    let diagnostic_config = MarkovSimulationVarianceConfig::new(
        80_401,
        1,
        12,
        6,
        LugsailConfig::over_lugsail_bartlett(),
        2,
        1024 * 1024,
    );
    let config = SaemConfig::new()
        .seed(80_400)
        .n_chains(2)
        .mcmc_iterations(1)
        .eta_block_iterations(1)
        .burn_in(0)
        .k1_iterations(2)
        .k2_iterations(4)
        .compute_map(true)
        .averaged_iterates(0.75)
        .markov_simulation_variance(diagnostic_config);
    let result = markov_output_problem()
        .fit_with(config)
        .expect("enabled Markov output fixture should fit");
    assert_eq!(result.termination_reason(), Some(&StopReason::MaxCycles));
    assert!(!result.conditional_modes().is_empty());
    let diagnostic = result.markov_simulation_variance();
    assert_eq!(diagnostic.config, Some(diagnostic_config));
    assert!(!diagnostic.chains.is_empty());
    assert!(!diagnostic.lambda.is_empty());
    assert!(!diagnostic.xi.is_empty());
    assert!(!diagnostic.simulation_covariance.is_empty());

    let directory = temp_output_dir("markov-schema-six");
    result
        .write_outputs(&directory, 0.0, 0.0)
        .expect("enabled Markov output should write");
    let record = ParametricResultRecord::read_json(directory.join("result.json"))
        .expect("enabled schema-eight result should load");
    assert_eq!(record.schema_version, 9);
    assert_eq!(
        record.config.estimator_policy,
        SaemEstimatorPolicy::AveragedIterates { alpha: 0.75 }
    );
    assert_eq!(
        record.config.markov_simulation_variance,
        Some(diagnostic_config)
    );
    let loaded = &record.markov_simulation_variance;
    assert_eq!(loaded.config, diagnostic.config);
    assert_eq!(loaded.coordinates, diagnostic.coordinates);
    assert_eq!(loaded.chain_count, diagnostic.chain_count);
    assert_eq!(loaded.n_avg, diagnostic.n_avg);
    assert_json_close(
        &serde_json::to_value(&loaded.rank_diagnostics).expect("serialize loaded rank diagnostics"),
        &serde_json::to_value(&diagnostic.rank_diagnostics)
            .expect("serialize in-memory rank diagnostics"),
    );
    assert_eq!(loaded.status, diagnostic.status);
    assert_eq!(loaded.lambda_status, diagnostic.lambda_status);
    assert_eq!(loaded.xi_status, diagnostic.xi_status);
    assert_eq!(
        loaded.simulation_covariance_status,
        diagnostic.simulation_covariance_status
    );
    let assert_matrix_close = |actual: &[Vec<f64>], expected: &[Vec<f64>]| {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_float_slice_close(actual, expected);
        }
    };
    assert_matrix_close(&loaded.lambda, &diagnostic.lambda);
    assert_matrix_close(&loaded.xi, &diagnostic.xi);
    assert_matrix_close(
        &loaded.simulation_covariance,
        &diagnostic.simulation_covariance,
    );
    assert_eq!(loaded.chains.len(), diagnostic.chains.len());
    for (actual, expected) in loaded.chains.iter().zip(&diagnostic.chains) {
        assert_eq!(actual.chain, expected.chain);
        assert_eq!(actual.status, expected.status);
        assert_eq!(actual.proposals, expected.proposals);
        assert_eq!(actual.accepts, expected.accepts);
        assert_eq!(actual.state_changes, expected.state_changes);
        assert_matrix_close(&actual.bm_batch, &expected.bm_batch);
        assert_matrix_close(&actual.bm_batch_over_r, &expected.bm_batch_over_r);
        assert_matrix_close(&actual.lugsail_lrv, &expected.lugsail_lrv);
    }
    let rows: Vec<_> = record
        .tables
        .statistics
        .iter()
        .filter(|row| row.kind.starts_with("markov_"))
        .collect();
    assert!(rows.iter().any(|row| row.kind == "markov_status"));
    assert!(rows.iter().any(|row| row.kind == "markov_config"));
    assert!(rows.iter().any(|row| row.kind == "markov_lugsail_lrv"));
    assert!(rows.iter().all(|row| row
        .status
        .as_deref()
        .is_some_and(|status| !status.is_empty())));

    record
        .warm_start_problem(markov_output_equation(), markov_output_data())
        .expect("enabled schema-six result should warm start");
    fs::remove_dir_all(directory).expect("temporary output directory should be removable");
}
