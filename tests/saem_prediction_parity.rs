use pharmsol::{prelude::Prediction, Predictions};
use pmcore::prelude::*;

const EXPECTED_PREDICTIONS: [f64; 5] = [
    1.902_458_849_001_428,
    1.809_674_836_071_919,
    1.637_461_506_155_963_6,
    1.340_640_092_071_278_7,
    0.898_657_928_234_443_1,
];
const EXPECTED_CONDITIONAL_NLL: f64 = -2.233_675_005_800_264;
const EXPECTED_N2LL: f64 = -4.467_350_011_600_528;

fn analytical_model() -> pharmsol::equation::Analytical {
    analytical! {
        name: "saem_v07_one_compartment",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [
            bolus(iv_bolus) -> central,
        ],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    }
}

fn raw_ode_model() -> pharmsol::equation::ODE {
    ode! {
        name: "saem_v07_one_compartment",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [
            bolus(iv_bolus) -> central,
        ],
        diffeq: |x, _t, dx| {
            dx[central] = -ke * x[central];
        },
        out: |x, _t, y| {
            y[cp] = x[central] / v;
        },
    }
}

fn ode_model() -> pharmsol::equation::ODE {
    raw_ode_model()
        .with_solver(pharmsol::equation::OdeSolver::Bdf)
        .with_tolerances(1e-8, 1e-10)
}

fn data() -> Data {
    Data::new(vec![Subject::builder("v07")
        .bolus(0.0, 100.0, "iv_bolus")
        .observation(0.5, 1.9, "cp")
        .observation(1.0, 1.8, "cp")
        .observation(2.0, 1.6, "cp")
        .observation(4.0, 1.3, "cp")
        .observation(8.0, 0.8, "cp")
        .build()])
}

fn config() -> SaemConfig {
    SaemConfig::new()
        .seed(20_260_707)
        .n_chains(2)
        .mcmc_iterations(1)
        .burn_in(1)
        .k1_iterations(1)
        .k2_iterations(0)
        .compute_map(false)
}

fn within_d1(actual: f64, expected: f64) -> bool {
    let absolute = (actual - expected).abs();
    let relative = absolute / expected.abs().max(f64::MIN_POSITIVE);
    absolute <= 1e-6 || relative <= 1e-4
}

#[test]
fn explicit_bdf_estimation_profiles_meet_analytical_oracle_bounds() {
    let prediction_data = data();
    let subject = &prediction_data.subjects()[0];
    let analytical = analytical_model()
        .estimate_predictions_dense(subject, &[0.1, 50.0])
        .expect("analytical predictions should succeed")
        .get_predictions();
    let release_profile = ode_model()
        .estimate_predictions_dense(subject, &[0.1, 50.0])
        .expect("release-profile ODE predictions should succeed")
        .get_predictions();
    let oracle_profile = raw_ode_model()
        .with_solver(pharmsol::equation::OdeSolver::Bdf)
        .with_tolerances(1e-10, 1e-12)
        .estimate_predictions_dense(subject, &[0.1, 50.0])
        .expect("oracle-profile ODE predictions should succeed")
        .get_predictions();

    assert_eq!(release_profile.len(), analytical.len());
    assert_eq!(oracle_profile.len(), analytical.len());
    let maximum_error = |predictions: &[Prediction]| {
        predictions
            .iter()
            .zip(&analytical)
            .map(|(numerical, exact)| (numerical.prediction() - exact.prediction()).abs())
            .fold(0.0_f64, f64::max)
    };
    let release_error = maximum_error(&release_profile);
    let oracle_error = maximum_error(&oracle_profile);

    assert!(
        release_error <= 1e-7,
        "release-profile error={release_error}"
    );
    assert!(oracle_error <= 1e-8, "oracle-profile error={oracle_error}");
}

#[test]
fn deterministic_analytical_and_ode_predictions_and_objectives_match() {
    let analytical = analytical_model();
    let ode = ode_model();
    let prediction_data = data();
    let subject = &prediction_data.subjects()[0];

    let analytical_predictions = analytical
        .estimate_predictions_dense(subject, &[0.1, 50.0])
        .expect("analytical predictions should succeed")
        .get_predictions();
    let ode_predictions = ode
        .estimate_predictions_dense(subject, &[0.1, 50.0])
        .expect("ODE predictions should succeed")
        .get_predictions();

    assert_eq!(analytical_predictions.len(), EXPECTED_PREDICTIONS.len());
    assert_eq!(ode_predictions.len(), EXPECTED_PREDICTIONS.len());
    for ((analytical_prediction, ode_prediction), expected) in analytical_predictions
        .iter()
        .zip(&ode_predictions)
        .zip(EXPECTED_PREDICTIONS)
    {
        let analytical_value = analytical_prediction.prediction();
        let ode_value = ode_prediction.prediction();
        assert!(
            (analytical_value - expected).abs() <= 1e-10,
            "analytical={analytical_value:.16}, expected={expected:.16}",
        );
        assert!(
            within_d1(ode_value, expected),
            "ODE={ode_value:.16}, expected={expected:.16}",
        );
        assert!(
            (ode_value - analytical_value).abs() <= 1e-7,
            "release-profile ODE={ode_value:.16}, analytical={analytical_value:.16}",
        );
        assert!(within_d1(ode_value, analytical_value));
    }

    let analytical_result = EstimationProblem::parametric(analytical, data())
        .parameter(
            Parameter::log("ke")
                .with_initial(0.1)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::log("v")
                .with_initial(50.0)
                .fixed()
                .without_random_effect(),
        )
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.25)).fixed(),
        )
        .build()
        .expect("analytical problem should build")
        .fit_with(config())
        .expect("analytical fit should complete");
    let ode_result = EstimationProblem::parametric(ode, data())
        .parameter(
            Parameter::log("ke")
                .with_initial(0.1)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::log("v")
                .with_initial(50.0)
                .fixed()
                .without_random_effect(),
        )
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.25)).fixed(),
        )
        .build()
        .expect("ODE problem should build")
        .fit_with(config())
        .expect("ODE fit should complete");

    assert!(
        (analytical_result.conditional_negative_log_likelihood() - EXPECTED_CONDITIONAL_NLL).abs()
            <= 1e-10
    );
    assert!((analytical_result.conditional_n2ll() - EXPECTED_N2LL).abs() <= 1e-10);
    assert!(within_d1(
        ode_result.conditional_negative_log_likelihood(),
        EXPECTED_CONDITIONAL_NLL,
    ));
    assert!(within_d1(ode_result.conditional_n2ll(), EXPECTED_N2LL));
    assert!(within_d1(
        ode_result.conditional_negative_log_likelihood(),
        analytical_result.conditional_negative_log_likelihood(),
    ));
    assert!(within_d1(
        ode_result.conditional_n2ll(),
        analytical_result.conditional_n2ll(),
    ));
    assert_eq!(
        analytical_result.termination_reason(),
        Some(&pmcore::algorithms::StopReason::MaxCycles),
    );
    assert_eq!(
        ode_result.termination_reason(),
        Some(&pmcore::algorithms::StopReason::MaxCycles),
    );
}

fn assert_finite_symmetric_information<E: Equation>(result: &ParametricResult<E>) {
    let information = result.information_diagnostics();
    assert_eq!(information.recursion_cycles, 1);
    assert!(!information.coordinates.is_empty());
    for matrix in [
        &information.g,
        &information.expected_complete_hessian,
        &information.observed_hessian,
        &information.observed_information,
    ] {
        assert!(matrix.iter().flatten().all(|value| value.is_finite()));
        for (row_index, row) in matrix.iter().enumerate() {
            for (column_index, column) in matrix.iter().take(row_index).enumerate() {
                assert!((row[column_index] - column[row_index]).abs() < 1e-12);
            }
        }
    }
    assert_eq!(
        result.termination_reason(),
        Some(&pmcore::algorithms::StopReason::MaxCycles)
    );
}

#[test]
fn analytical_and_ode_fits_produce_finite_symmetric_information_diagnostics() {
    let diagnostic_config = SaemConfig::new()
        .seed(20_260_707)
        .n_chains(2)
        .mcmc_iterations(1)
        .burn_in(1)
        .k1_iterations(2)
        .k2_iterations(0)
        .compute_map(false);
    let analytical = EstimationProblem::parametric(analytical_model(), data())
        .parameter(
            Parameter::log("ke")
                .with_initial(0.1)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::log("v")
                .with_initial(50.0)
                .fixed()
                .without_random_effect(),
        )
        .error_model("cp", ResidualErrorModel::constant(0.25))
        .build()
        .unwrap()
        .fit_with(diagnostic_config.clone())
        .unwrap();
    let ode = EstimationProblem::parametric(ode_model(), data())
        .parameter(
            Parameter::log("ke")
                .with_initial(0.1)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::log("v")
                .with_initial(50.0)
                .fixed()
                .without_random_effect(),
        )
        .error_model("cp", ResidualErrorModel::constant(0.25))
        .build()
        .unwrap()
        .fit_with(diagnostic_config)
        .unwrap();

    assert_finite_symmetric_information(&analytical);
    assert_finite_symmetric_information(&ode);

    let analytical_information = analytical.information_diagnostics();
    let ode_information = ode.information_diagnostics();
    assert_eq!(
        analytical_information.coordinates, ode_information.coordinates,
        "analytical and ODE information coordinates must have identical order"
    );
    assert_eq!(analytical_information.status, ode_information.status);
    for (analytical_value, ode_value) in analytical_information
        .delta
        .iter()
        .zip(&ode_information.delta)
    {
        assert!(within_d1(*analytical_value, *ode_value));
    }
    for (analytical_matrix, ode_matrix) in [
        (&analytical_information.g, &ode_information.g),
        (
            &analytical_information.expected_complete_hessian,
            &ode_information.expected_complete_hessian,
        ),
        (
            &analytical_information.observed_hessian,
            &ode_information.observed_hessian,
        ),
        (
            &analytical_information.observed_information,
            &ode_information.observed_information,
        ),
    ] {
        for (analytical_value, ode_value) in analytical_matrix
            .iter()
            .flatten()
            .zip(ode_matrix.iter().flatten())
        {
            assert!(within_d1(*analytical_value, *ode_value));
        }
    }
}
