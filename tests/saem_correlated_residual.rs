use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use pharmsol::prelude::*;
use pmcore::prelude::*;
use pmcore::results::{CovarianceCycleUpdateOutcome, InformationCoordinateKind};

fn equation() -> pharmsol::equation::Analytical {
    analytical! {
        name: "n8_correlated_residual_fixture",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [infusion(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| { y[cp] = x[central] / v; },
    }
}

fn data() -> Data {
    Data::new(vec![
        Subject::builder("n8-1")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 4.8, "cp")
            .observation(3.0, 3.0, "cp")
            .observation(6.0, 1.5, "cp")
            .reset()
            .infusion(12.0, 120.0, "iv", 0.5)
            .observation(13.0, 5.7, "cp")
            .observation(15.0, 3.5, "cp")
            .observation(18.0, 1.8, "cp")
            .build(),
        Subject::builder("n8-2")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 5.2, "cp")
            .observation(3.0, 3.3, "cp")
            .observation(6.0, 1.7, "cp")
            .reset()
            .infusion(12.0, 120.0, "iv", 0.5)
            .observation(13.0, 6.1, "cp")
            .observation(15.0, 3.8, "cp")
            .observation(18.0, 2.0, "cp")
            .build(),
        Subject::builder("n8-3")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 4.5, "cp")
            .observation(3.0, 2.8, "cp")
            .observation(6.0, 1.4, "cp")
            .reset()
            .infusion(12.0, 120.0, "iv", 0.5)
            .observation(13.0, 5.4, "cp")
            .observation(15.0, 3.3, "cp")
            .observation(18.0, 1.6, "cp")
            .build(),
        Subject::builder("n8-4")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 5.0, "cp")
            .observation(3.0, 3.1, "cp")
            .observation(6.0, 1.6, "cp")
            .reset()
            .infusion(12.0, 120.0, "iv", 0.5)
            .observation(13.0, 5.9, "cp")
            .observation(15.0, 3.6, "cp")
            .observation(18.0, 1.9, "cp")
            .build(),
    ])
}

fn problem() -> EstimationProblem<pharmsol::equation::Analytical, Parametric> {
    EstimationProblem::parametric(equation(), data())
        .parameter(Parameter::log("ke").with_initial(0.25).fixed())
        .parameter(
            Parameter::log("v")
                .with_initial(20.0)
                .fixed()
                .without_random_effect(),
        )
        .omega(Omega::diagonal([("ke", 0.05)]))
        .iov(Iov::diagonal([("ke", 0.03)]))
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::correlated_combined(0.3, 0.12, 0.1))
                .fixed_correlated_combined_additive(),
        )
        .build()
        .expect("N8 analytical problem")
}

fn config() -> SaemConfig {
    SaemConfig::new()
        .seed(0x6e38_2026)
        .n_chains(2)
        .mcmc_iterations(1)
        .burn_in(0)
        .k1_iterations(2)
        .k2_iterations(1)
        .compute_map(false)
}

fn scalar_nll(residual: f64, variance: f64) -> f64 {
    0.5 * (variance.ln() + residual * residual / variance)
}

fn pair_nll(residual: [f64; 2], omega: f64, variances: [f64; 2]) -> f64 {
    let a = omega + variances[0];
    let b = omega;
    let d = omega + variances[1];
    let determinant = a * d - b * b;
    let quadratic = (d * residual[0] * residual[0] - 2.0 * b * residual[0] * residual[1]
        + a * residual[1] * residual[1])
        / determinant;
    0.5 * (determinant.ln() + quadratic)
}

#[test]
fn declaration_domains_are_strict_and_fixed_free_controls_are_independent() {
    for model in [
        ResidualErrorModel::correlated_combined(0.0, 0.1, 0.0),
        ResidualErrorModel::correlated_combined(0.1, 0.0, 0.0),
        ResidualErrorModel::correlated_combined(0.1, 0.1, -1.0),
        ResidualErrorModel::correlated_combined(0.1, 0.1, 1.0),
        ResidualErrorModel::correlated_combined(0.1, 0.1, f64::NAN),
    ] {
        let built = EstimationProblem::parametric(equation(), data())
            .parameter(Parameter::log("ke").with_initial(0.25).fixed())
            .parameter(
                Parameter::log("v")
                    .with_initial(20.0)
                    .fixed()
                    .without_random_effect(),
            )
            .omega(Omega::new().fixed_variance("ke", 0.05))
            .error_model("cp", model)
            .build();
        assert!(
            built.is_err(),
            "invalid declaration was accepted: {model:?}"
        );
    }

    for free in [
        [true, true, true],
        [false, true, true],
        [true, false, true],
        [true, true, false],
        [false, false, true],
        [false, true, false],
        [true, false, false],
        [false, false, false],
    ] {
        let declaration =
            ParametricErrorModel::new(ResidualErrorModel::correlated_combined(0.3, 0.12, -0.2))
                .with_correlated_combined_additive_estimate(free[0])
                .with_correlated_combined_proportional_estimate(free[1])
                .with_correlated_combined_correlation_estimate(free[2]);
        let built = EstimationProblem::parametric(equation(), data())
            .parameter(Parameter::log("ke").with_initial(0.25).fixed())
            .parameter(
                Parameter::log("v")
                    .with_initial(20.0)
                    .fixed()
                    .without_random_effect(),
            )
            .omega(Omega::new().fixed_variance("ke", 0.05))
            .error_model("cp", declaration)
            .build();
        assert!(built.is_ok(), "fixed/free mask {free:?}");
    }
}

#[test]
fn iov_and_residual_identifiability_is_design_dependent() {
    let residual = 0.4;
    assert_eq!(
        scalar_nll(residual, 0.2 + 0.5),
        scalar_nll(residual, 0.3 + 0.4)
    );

    let repeated_residual = [0.4, -0.2];
    let first = pair_nll(repeated_residual, 0.2, [0.5, 0.5]);
    let second = pair_nll(repeated_residual, 0.3, [0.4, 0.4]);
    assert!((first - second).abs() > 1e-6);

    let truth = [0.7_f64, 0.25_f64, -0.3_f64];
    let one_prediction = 2.0_f64;
    let one_variance = truth[0].powi(2)
        + 2.0 * truth[2] * truth[0] * truth[1] * one_prediction
        + truth[1].powi(2) * one_prediction.powi(2);
    let alternative_a = 0.5_f64;
    let alternative_b = 0.3_f64;
    let alternative_rho =
        (one_variance - alternative_a.powi(2) - alternative_b.powi(2) * one_prediction.powi(2))
            / (2.0 * alternative_a * alternative_b * one_prediction);
    assert!(alternative_rho > -1.0 && alternative_rho < 1.0);
    let alternative_variance = alternative_a.powi(2)
        + 2.0 * alternative_rho * alternative_a * alternative_b * one_prediction
        + alternative_b.powi(2) * one_prediction.powi(2);
    assert!((one_variance - alternative_variance).abs() < 1e-14);

    let levels = [-1.0_f64, 0.0, 2.0];
    let design_determinant =
        (levels[1] - levels[0]) * (levels[2] - levels[0]) * (levels[2] - levels[1]);
    assert!(design_determinant.abs() > 0.0);
    assert!(levels.into_iter().any(|prediction| {
        let true_variance = truth[0].powi(2)
            + 2.0 * truth[2] * truth[0] * truth[1] * prediction
            + truth[1].powi(2) * prediction.powi(2);
        let other_variance = alternative_a.powi(2)
            + 2.0 * alternative_rho * alternative_a * alternative_b * prediction
            + alternative_b.powi(2) * prediction.powi(2);
        (true_variance - other_variance).abs() > 1e-6
    }));
}

#[test]
fn short_iiv_iov_fit_routes_components_and_schema_nine_roundtrips() {
    let result = problem().fit_with(config()).expect("N8 short fit");
    let residual = &result.residual_error_estimates()[0];
    let ResidualErrorModel::CorrelatedCombined { a, b, rho } = residual.model else {
        panic!("correlated-combined family was not retained")
    };
    assert_eq!(a, 0.3);
    assert!(b.is_finite() && b > 0.0);
    assert!(rho.is_finite() && rho > -1.0 && rho < 1.0);
    assert_eq!(residual.combined_additive_estimated, Some(false));
    assert_eq!(residual.combined_proportional_estimated, Some(true));
    assert_eq!(residual.correlation_estimated, Some(true));
    assert!(result.omega().iter().all(|value| value.is_finite()));
    assert!(result
        .omega_iov()
        .expect("IOV covariance")
        .iter()
        .all(|value| value.is_finite()));

    for cycle in result.cycle_diagnostics() {
        assert!(!matches!(
            cycle.omega_update.outcome,
            CovarianceCycleUpdateOutcome::NotAttempted { .. }
        ));
        assert!(!matches!(
            cycle.omega_iov_update.outcome,
            CovarianceCycleUpdateOutcome::NotAttempted { .. }
        ));
        let diagnostic = cycle.residual_diagnostic("cp").unwrap();
        assert_eq!(diagnostic.prediction_evaluation_count, 48);
        assert!(diagnostic.optimizer_objective.is_some());
        assert!(diagnostic.optimizer_iterations.is_some());
        assert!(!diagnostic.update_rejected);
    }

    let tables = result.tables(0.0, 0.0).unwrap();
    assert_eq!(
        tables
            .residual_error
            .iter()
            .map(|row| (row.component.as_str(), row.estimated))
            .collect::<Vec<_>>(),
        [
            ("additive", false),
            ("proportional", true),
            ("correlation", true)
        ]
    );
    assert!(result
        .information_diagnostics()
        .coordinates
        .iter()
        .any(|coordinate| {
            matches!(
                &coordinate.kind,
                InformationCoordinateKind::Residual {
                    component,
                    ..
                } if component == "correlation"
            )
        }));

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!("pmcore-n8-{unique}"));
    result.write_outputs(&directory, 0.0, 0.0).unwrap();
    let path = directory.join("result.json");
    let record = ParametricResultRecord::read_json(&path).unwrap();
    assert_eq!(record.schema_version, 9);
    assert_eq!(record.tables.residual_error, tables.residual_error);

    let live_warm = result
        .warm_start_problem()
        .unwrap()
        .fit_with(config())
        .unwrap();
    let persisted_warm = record
        .warm_start_problem(equation(), data())
        .unwrap()
        .fit_with(config())
        .unwrap();
    for warm in [&live_warm, &persisted_warm] {
        let ResidualErrorModel::CorrelatedCombined { a, b, rho } =
            warm.residual_error_estimates()[0].model
        else {
            panic!("warm start changed residual family")
        };
        assert_eq!(a, 0.3);
        assert!(b.is_finite() && b > 0.0);
        assert!(rho.is_finite() && rho > -1.0 && rho < 1.0);
    }

    let mut tampered: serde_json::Value =
        serde_json::from_reader(fs::File::open(&path).unwrap()).unwrap();
    tampered["source_metadata"]["residual_outputs"][0]["values"][2] = serde_json::json!(1.0);
    let tampered_path = directory.join("tampered.json");
    fs::write(
        &tampered_path,
        serde_json::to_vec_pretty(&tampered).unwrap(),
    )
    .unwrap();
    assert!(ParametricResultRecord::read_json(&tampered_path).is_err());
    fs::remove_dir_all(directory).unwrap();
}
