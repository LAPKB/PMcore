use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

use pharmsol::prelude::*;
use pmcore::prelude::*;
use pmcore::results::{InformationStatus, ParametricResultRecord};

fn direct_equation() -> pharmsol::equation::Analytical {
    analytical! {
        name: "n8b_direct",
        params: [ke, v], states: [central], outputs: [cp],
        routes: [infusion(iv) -> central], structure: one_compartment,
        out: |_x, _p, _t, _cov, y| { y[cp] = v; },
    }
}

fn direct_data(center: f64) -> Data {
    Data::new(vec![Subject::builder("direct")
        .observation(1.0, center - 1.0, "cp")
        .observation(1.0, center + 1.0, "cp")
        .build()])
}

fn fixed_error() -> ParametricErrorModel {
    ParametricErrorModel::new(ResidualErrorModel::constant(1.0)).fixed()
}

fn direct_problem(initial: f64) -> EstimationProblem<pharmsol::equation::Analytical, Parametric> {
    EstimationProblem::parametric(direct_equation(), direct_data(20.0))
        .parameter(
            Parameter::log("ke")
                .with_initial(0.25)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::log("v")
                .with_initial(initial)
                .without_random_effect(),
        )
        .error_model("cp", fixed_error())
        .build()
        .unwrap()
}

fn config() -> SaemConfig {
    SaemConfig::new()
        .seed(0x6e38_b026)
        .n_chains(1)
        .mcmc_iterations(1)
        .burn_in(0)
        .k1_iterations(1)
        .k2_iterations(0)
        .compute_map(false)
}

#[test]
fn activation_waits_for_annealing_boundary_and_applies_one_gain() {
    let mut delayed = config().burn_in(1).k1_iterations(2).k2_iterations(3);
    delayed.sa_iterations = 4;
    let result = direct_problem(40.0).fit_with(delayed).unwrap();
    let cycles = result.cycle_diagnostics();
    assert_eq!(cycles.len(), 5);
    for cycle in &cycles[..3] {
        assert!((cycle.population_parameters[1] - 40.0).abs() < 1e-12);
    }
    assert_eq!(cycles[3].iteration, 4);
    assert_eq!(cycles[3].stochastic_approximation_step, 0.5);
    let expected = (40.0_f64 * 20.0).sqrt();
    assert!((cycles[3].population_parameters[1] - expected).abs() < 1e-3);
}

#[test]
fn tiny_no_iiv_fixture_uses_pre_update_residual_evidence_then_next_cycle_state() {
    let result = EstimationProblem::parametric(direct_equation(), direct_data(20.0))
        .parameter(
            Parameter::log("ke")
                .with_initial(0.25)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::real("v")
                .with_initial(40.0)
                .without_random_effect(),
        )
        .error_model("cp", ResidualErrorModel::constant(5.0))
        .build()
        .unwrap()
        .fit_with(config().k1_iterations(2))
        .unwrap();
    let cycles = result.cycle_diagnostics();
    assert_eq!(cycles.len(), 2);
    assert!((cycles[0].population_parameters[1] - 20.0).abs() < 1e-3);
    let ResidualErrorModel::Constant { a: first_sigma } =
        cycles[0].residual_error_estimates[0].model
    else {
        panic!("constant residual model was not retained")
    };
    assert!((first_sigma - 401.0_f64.sqrt()).abs() < 1e-6);
    let ResidualErrorModel::Constant { a: second_sigma } =
        cycles[1].residual_error_estimates[0].model
    else {
        panic!("constant residual model was not retained")
    };
    assert!((second_sigma - 1.0).abs() < 1e-6);
    assert!((result.population_parameters()[1] - 20.0).abs() < 1e-3);
}

fn mixed_equation() -> pharmsol::equation::Analytical {
    analytical! {
        name: "n8b_mixed",
        params: [ke, v], states: [central], outputs: [cp],
        routes: [infusion(iv) -> central], structure: one_compartment,
        out: |x, _p, _t, _cov, y| { y[cp] = x[central] / v; },
    }
}

#[test]
fn mixed_iiv_and_no_iiv_coordinates_follow_separate_paths() {
    let data = Data::new(vec![
        Subject::builder("m1")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 4.8, "cp")
            .observation(3.0, 3.0, "cp")
            .build(),
        Subject::builder("m2")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 5.2, "cp")
            .observation(3.0, 3.4, "cp")
            .build(),
    ]);
    let result = EstimationProblem::parametric(mixed_equation(), data)
        .parameter(Parameter::log("ke").with_initial(0.25))
        .parameter(
            Parameter::log("v")
                .with_initial(30.0)
                .without_random_effect(),
        )
        .omega(Omega::diagonal_variances([("ke", 0.05)]))
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.3)).fixed(),
        )
        .build()
        .unwrap()
        .fit_with(config())
        .unwrap();
    assert_eq!(result.random_effect_names(), ["ke"]);
    assert_eq!(result.omega().dim(), (1, 1));
    assert!(result
        .eta_chain_means()
        .iter()
        .all(|eta| eta.values.len() == 1));
    assert!(result
        .population_parameters()
        .iter()
        .all(|value| value.is_finite()));
    assert!(matches!(result.information_diagnostics().status,
        InformationStatus::Unsupported(ref reason) if reason.contains("estimated non-IIV")));
}

#[test]
fn zero_iiv_covariate_effect_is_estimated_jointly_with_its_intercept() {
    let subjects = [-1.0_f64, 0.0, 1.0]
        .into_iter()
        .enumerate()
        .map(|(index, wt)| {
            let prediction = (20.0_f64.ln() + 0.2 * wt).exp();
            Subject::builder(format!("c{index}"))
                .covariate("wt", 0.0, wt)
                .observation(1.0, prediction - 0.5, "cp")
                .observation(1.0, prediction + 0.5, "cp")
                .build()
        })
        .collect();
    let result = EstimationProblem::parametric(direct_equation(), Data::new(subjects))
        .parameter(
            Parameter::log("ke")
                .with_initial(0.25)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::log("v")
                .with_initial(18.0)
                .without_random_effect(),
        )
        .covariate_effect(CovariateEffect::continuous("v", "wt", 0.0).with_initial(0.0))
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.5)).fixed(),
        )
        .build()
        .unwrap()
        .fit_with(config())
        .unwrap();
    assert!((result.population_parameters()[1] - 20.0).abs() < 1e-3);
    assert!((result.covariates().unwrap().estimates()[0].estimate() - 0.2).abs() < 1e-3);
    assert!(result.random_effect_names().is_empty());
}

fn saturation_equation() -> pharmsol::equation::Analytical {
    analytical! {
        name: "n8b_invalid_trial",
        params: [ke, v], states: [central], outputs: [cp],
        routes: [infusion(iv) -> central], structure: one_compartment,
        out: |_x, _p, _t, _cov, y| { y[cp] = 1.0 + 0.0 * v; },
    }
}

#[test]
fn invalid_optimizer_trials_are_penalized_without_aborting_the_fit() {
    let result = EstimationProblem::parametric(saturation_equation(), direct_data(1.0))
        .parameter(
            Parameter::log("ke")
                .with_initial(0.25)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::log("v")
                .with_initial(f64::MAX / 2.0)
                .without_random_effect(),
        )
        .error_model("cp", fixed_error())
        .build()
        .unwrap()
        .fit_with(config())
        .unwrap();
    assert!(result.population_parameters()[1].is_finite());
    assert!(result.population_parameters()[1] > 0.0);
    assert_eq!(result.cycle_diagnostics().len(), 1);
}

#[test]
fn schema_nine_lifecycle_count_warm_start_and_information_are_honest() {
    let result = direct_problem(30.0)
        .fit_with(config().marginal_likelihood(MarginalLikelihoodConfig::new(16, 88, 5, 1.5)))
        .unwrap();
    assert_eq!(result.free_parameter_count(), 1);
    assert!(result.aic().unwrap().is_finite() && result.bic().unwrap().is_finite());
    assert!(matches!(result.information_diagnostics().status,
        InformationStatus::Unsupported(ref reason) if reason.contains("estimated non-IIV")));

    let path = std::env::temp_dir().join(format!(
        "pmcore-n8b-{}-{}.json",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    result.write_json(&path, 0.0, 0.0).unwrap();
    let record = ParametricResultRecord::read_json(&path).unwrap();
    assert_eq!(record.schema_version, 9);
    let warm = record
        .warm_start_problem(direct_equation(), direct_data(20.0))
        .unwrap();
    assert_eq!(
        warm.parameters().items[1].initial.unwrap().to_bits(),
        result.population_parameters()[1].to_bits()
    );
    fs::remove_file(path).unwrap();
}

#[test]
fn variance_and_sd_diagonal_constructors_are_explicit_and_fail_closed() {
    fn build(
        omega: Omega,
        iov: Iov,
    ) -> anyhow::Result<EstimationProblem<pharmsol::equation::Analytical, Parametric>> {
        EstimationProblem::parametric(direct_equation(), direct_data(20.0))
            .parameter(Parameter::log("ke").with_initial(0.25).fixed())
            .parameter(
                Parameter::log("v")
                    .with_initial(20.0)
                    .fixed()
                    .without_random_effect(),
            )
            .omega(omega)
            .iov(iov)
            .error_model("cp", fixed_error())
            .build()
    }
    let variance = build(
        Omega::diagonal_variances([("ke", 0.09)]),
        Iov::diagonal_variances([("v", 0.16)]),
    )
    .unwrap();
    let sd = build(
        Omega::diagonal_standard_deviations([("ke", 0.3)]),
        Iov::diagonal_standard_deviations([("v", 0.4)]),
    )
    .unwrap();
    assert!((variance.omega()[(0, 0)] - sd.omega()[(0, 0)]).abs() < 1e-15);
    assert!(
        (variance.omega_iov().unwrap()[(0, 0)] - sd.omega_iov().unwrap()[(0, 0)]).abs() < 1e-15
    );
    for invalid in [0.0, -1.0, f64::NAN, f64::MAX] {
        assert!(build(
            Omega::diagonal_standard_deviations([("ke", invalid)]),
            Iov::diagonal_variances([("v", 0.16)])
        )
        .is_err());
    }
}
