use super::*;
#[test]
fn e_step_rescores_chain_zero_parameters() {
    let mut state = SaemState::from_problem(problem(), &SaemConfig::new().n_chains(1)).unwrap();
    let initial = state.log_likelihood();

    state.etas[0][0][0] = 2.0_f64.ln();
    state.e_step().unwrap();

    assert!(state.log_likelihood().is_finite());
    assert_ne!(state.log_likelihood(), initial);
    assert_eq!(state.negative_log_likelihood(), -state.log_likelihood());
}

#[test]
fn iov_result_retains_named_omega_iov() {
    let result = iov_problem()
        .fit_with(
            SaemConfig::new()
                .n_chains(2)
                .k1_iterations(1)
                .k2_iterations(0)
                .burn_in(1),
        )
        .unwrap();

    assert_eq!(result.iov_effect_names(), &["ke"]);
    assert_eq!(result.omega_iov(), Some(&ndarray::array![[0.1]]));
    assert_eq!(result.conditional_modes().len(), 1);
    assert_eq!(result.conditional_modes()[0].kappas.len(), 2);
    assert!(result.conditional_modes()[0].objective.is_finite());
}

#[test]
fn result_reports_final_chain_means_for_eta_and_kappa() {
    let mut state = SaemState::from_problem(iov_problem(), &SaemConfig::new().n_chains(2)).unwrap();
    state.etas[0][0][0] = 0.2;
    state.etas[0][1][0] = 0.4;
    state.kappas[0][0][0][0] = -0.2;
    state.kappas[0][1][0][0] = 0.4;
    state.kappas[0][0][1][0] = 0.1;
    state.kappas[0][1][1][0] = 0.3;

    let result = Box::new(state).into_result().unwrap();

    assert_eq!(result.eta_chain_means().len(), 1);
    assert!((result.eta_chain_means()[0].values[0] - 0.3).abs() < 1e-12);
    assert_eq!(result.kappa_chain_means().len(), 2);
    assert_eq!(result.kappa_chain_means()[0].occasion_index, 0);
    assert!((result.kappa_chain_means()[0].values[0] - 0.1).abs() < 1e-12);
    assert_eq!(result.kappa_chain_means()[1].occasion_index, 1);
    assert!((result.kappa_chain_means()[1].values[0] - 0.2).abs() < 1e-12);
}

#[test]
fn result_retains_immutable_cycle_diagnostics() {
    let config = SaemConfig::new()
        .n_chains(1)
        .mcmc_iterations(1)
        .burn_in(1)
        .k1_iterations(1)
        .k2_iterations(1)
        .compute_map(false);
    let mut state = SaemState::from_problem(problem(), &config).unwrap();
    state.step().unwrap();
    state.step().unwrap();

    let result = Box::new(state).into_result().unwrap();

    assert_eq!(result.parameter_names(), ["ke", "v"]);
    assert_eq!(result.data().subjects().len(), 2);
    assert_eq!(
        result
            .equation()
            .metadata()
            .expect("retained equation metadata")
            .outputs()[0]
            .name(),
        "0"
    );
    assert_eq!(result.cycle_diagnostics().len(), 2);
    assert_eq!(result.cycle_diagnostics()[0].iteration, 1);
    assert_eq!(result.cycle_diagnostics()[0].phase, SaemPhase::BurnIn);
    assert_eq!(result.cycle_diagnostics()[1].iteration, 2);
    assert_eq!(result.cycle_diagnostics()[1].phase, SaemPhase::Smoothing);
    assert_eq!(
        result.cycle_diagnostics()[0].population_parameters,
        vec![0.2, 10.0]
    );
    let final_cycle = &result.cycle_diagnostics()[1];
    assert_eq!(
        final_cycle.population_parameters,
        result.population_parameters()
    );
    assert_eq!(&final_cycle.omega, result.omega());
    assert_eq!(final_cycle.omega_iov.as_ref(), result.omega_iov());
    assert_eq!(
        final_cycle.residual_error_estimates,
        result.residual_error_estimates()
    );
    assert!(final_cycle.conditional_negative_log_likelihood.is_finite());
    assert!(final_cycle.eta_log_prior.is_finite());
    assert!(final_cycle.kappa_log_prior.is_finite());
}

#[test]
fn conditional_modes_can_be_disabled_without_relabeling_chain_means() {
    let result = problem()
        .fit_with(
            SaemConfig::new()
                .n_chains(2)
                .k1_iterations(1)
                .k2_iterations(0)
                .burn_in(1)
                .compute_map(false),
        )
        .unwrap();

    assert!(result.conditional_modes().is_empty());
    assert_eq!(result.eta_chain_means().len(), 2);
    let error = result.conditional_predictions(0.25, 0.0).unwrap_err();
    assert_eq!(
        error.to_string(),
        "conditional predictions require conditional modes; rerun with compute_map(true)"
    );
}

#[test]
fn population_uncertainty_wires_analytical_fit_summary_without_changing_estimates() {
    let equation = analytical! {
        name: "population_uncertainty_summary_fixture",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [infusion(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| { y[cp] = x[central] / v; },
    };
    let data = Data::new(vec![
        Subject::builder("uncertainty-1")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 4.8, "cp")
            .observation(3.0, 3.0, "cp")
            .build(),
        Subject::builder("uncertainty-2")
            .infusion(0.0, 120.0, "iv", 0.5)
            .observation(1.0, 5.4, "cp")
            .observation(3.0, 3.2, "cp")
            .build(),
    ]);
    let problem = EstimationProblem::parametric(equation, data)
        .parameter(Parameter::log("ke").with_initial(0.25))
        .parameter(
            Parameter::log("v")
                .with_initial(20.0)
                .fixed()
                .without_random_effect(),
        )
        .omega(Omega::new().fixed_variance("ke", 0.09))
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.4)).fixed(),
        )
        .build()
        .expect("population uncertainty analytical fixture");
    let mut result = problem
        .fit_with(
            SaemConfig::new()
                .seed(0x6a_2026)
                .n_chains(2)
                .mcmc_iterations(1)
                .burn_in(1)
                .k1_iterations(1)
                .k2_iterations(0)
                .compute_map(false),
        )
        .expect("population uncertainty analytical fit");
    let estimates_before = result.population_parameters().to_vec();
    let objective_before = result.objf();
    assert_eq!(estimates_before, vec![0.25, 20.0]);
    assert_eq!(result.estimated_parameters(), &[true, false]);
    assert_eq!(
        result.population_uncertainty(),
        &derive_population_uncertainty(result.information_diagnostics())
    );

    let coordinates = result.information_diagnostics().coordinates.clone();
    assert_eq!(coordinates.len(), 1);
    assert_eq!(
        coordinates[0].kind,
        InformationCoordinateKind::Population { parameter_index: 0 }
    );
    result.population_uncertainty = PopulationUncertaintyDiagnostics {
        coordinates,
        free_covariance: Some(vec![vec![0.04]]),
        free_standard_errors: Some(vec![0.2]),
        spectral_condition_number: Some(1.0),
        status: PopulationUncertaintyStatus::Available,
        regularization: PopulationUncertaintyRegularization::None,
    };

    let summary = result.population_summary();
    assert_eq!(result.population_parameters(), estimates_before);
    assert_eq!(result.objf().to_bits(), objective_before.to_bits());
    assert_eq!(
        summary
            .parameters
            .iter()
            .map(|parameter| parameter.estimate)
            .collect::<Vec<_>>(),
        estimates_before
    );
    assert!(
        (summary.parameters[0]
            .sd
            .expect("free log-scale parameter SE")
            - 0.2 * estimates_before[0])
            .abs()
            < 1e-12
    );
    assert!(
        (summary.parameters[0]
            .cv_percent
            .expect("free log-scale parameter CV")
            - 20.0)
            .abs()
            < 1e-12
    );
    assert_eq!(summary.parameters[1].sd, None);
    assert_eq!(summary.parameters[1].cv_percent, None);
}

#[test]
fn initialization_result_is_non_converged_snapshot() {
    let config = SaemConfig::new()
        .k1_iterations(1)
        .k2_iterations(1)
        .burn_in(1);
    let result = problem().fit_with(config).unwrap();
    let summary = result.summary();

    assert!(!result.converged());
    assert_eq!(result.termination_reason(), Some(&StopReason::MaxCycles));
    assert_ne!(result.termination_reason(), Some(&StopReason::Aborted));
    assert_ne!(
        result.termination_reason(),
        Some(&StopReason::NumericalFailure)
    );
    assert_eq!(result.iterations(), 2);
    assert_eq!(summary.subject_count, 2);
    assert_eq!(summary.observation_count, 4);
    assert_eq!(summary.parameter_count, 2);
    assert!(result.objf().is_finite());
    assert_eq!(result.population_parameters().len(), 2);
    assert_eq!(result.random_effect_names(), &["ke", "v"]);
    assert_eq!(result.omega().dim(), (2, 2));
    assert_eq!(result.residual_sigmas().len(), 1);
    assert_eq!(result.eta_chain_means().len(), 2);
    assert!(result.kappa_chain_means().is_empty());
    assert_eq!(result.conditional_modes().len(), 2);
    assert!(result
        .conditional_modes()
        .iter()
        .all(|mode| mode.objective.is_finite()));
    assert_eq!(result.population_summary().parameters.len(), 2);
    assert_eq!(result.individual_summaries().len(), 2);
}

// ─── Operational convergence tests ───────────────────────────────────

#[test]
fn operational_convergence_disabled_when_config_is_none() {
    use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
    let markov = MarkovSimulationVarianceConfig::new(
        7,
        0,
        12,
        6,
        LugsailConfig::over_lugsail_bartlett(),
        4,
        1024 * 1024,
    );
    let config = SaemConfig::new()
        .k1_iterations(2)
        .k2_iterations(2)
        .burn_in(0)
        .averaged_iterates(0.75)
        .markov_simulation_variance(markov)
        .compute_map(false)
        .seed(42);
    let result = problem().fit_with(config).unwrap();
    let ops = result.operational_diagnostics();
    assert!(ops.checks.is_empty());
    assert!(!ops.used_for_termination);
    assert!(!ops.final_check_reused);
    assert_eq!(result.termination_reason(), Some(&StopReason::MaxCycles));
}

#[test]
fn operational_convergence_records_checkpoints_when_configured() {
    use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
    let markov = MarkovSimulationVarianceConfig::new(
        7,
        0,
        12,
        6,
        LugsailConfig::over_lugsail_bartlett(),
        4,
        1024 * 1024,
    );
    let oc = OperationalConvergenceConfig::literature_guided(1, 1, 1.0, 0.95, 0.1, 0.02);
    let config = SaemConfig::new()
        .k1_iterations(1)
        .k2_iterations(3)
        .burn_in(0)
        .averaged_iterates(0.75)
        .markov_simulation_variance(markov)
        .covariance_stability(CovarianceStabilityConfig::new(0.01, 2))
        .operational_convergence(oc)
        .compute_map(false)
        .seed(43);
    let result = problem().fit_with(config).unwrap();
    let ops = result.operational_diagnostics();
    // Should have at least one checkpoint (smoothing phase produces checkpoints)
    assert!(!ops.checks.is_empty(), "expected at least one checkpoint");
    // Each checkpoint should have all fields populated
    for check in &ops.checks {
        assert!(check.checkpoint_seed.is_some());
        assert!(check.z_quantile.is_some());
        assert!(check.implied_minimum_ess.is_some());
        assert!(!check.criteria.is_empty());
        assert!(check.markov.is_some());
        assert_eq!(
            check.averaged_iterations,
            check.markov.as_ref().unwrap().n_avg
        );
    }
}

#[test]
fn operational_convergence_has_exact_criterion_names() {
    use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
    let markov = MarkovSimulationVarianceConfig::new(
        7,
        0,
        12,
        6,
        LugsailConfig::over_lugsail_bartlett(),
        4,
        1024 * 1024,
    );
    let oc = OperationalConvergenceConfig::literature_guided(1, 1, 1.0, 0.95, 0.1, 0.02);
    let config = SaemConfig::new()
        .k1_iterations(1)
        .k2_iterations(3)
        .burn_in(0)
        .averaged_iterates(0.75)
        .markov_simulation_variance(markov)
        .covariance_stability(CovarianceStabilityConfig::new(0.01, 2))
        .operational_convergence(oc)
        .compute_map(false)
        .seed(44);
    let result = problem().fit_with(config).unwrap();
    let ops = result.operational_diagnostics();
    assert!(!ops.checks.is_empty());
    let first_check = &ops.checks[0];
    let names: Vec<&str> = first_check
        .criteria
        .iter()
        .map(|c| c.name.as_str())
        .collect();
    assert!(names.contains(&"max_rhat"));
    assert!(names.contains(&"min_bulk_ess"));
    assert!(names.contains(&"min_average_bulk_ess_per_split_chain"));
    assert!(names.contains(&"relative_fixed_width"));
    assert!(names.contains(&"newton_displacement"));
    assert!(names.contains(&"newton_displacement_mc_sd"));
    assert!(names.contains(&"omega_boundary_rejection_run"));
    assert!(names.contains(&"omega_iov_boundary_rejection_run"));
}

#[test]
fn covariance_boundary_rejection_blocks_converged_stop_reason() {
    use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
    let markov = MarkovSimulationVarianceConfig::new(
        7,
        0,
        12,
        6,
        LugsailConfig::over_lugsail_bartlett(),
        4,
        1024 * 1024,
    );
    let oc = OperationalConvergenceConfig::literature_guided(1, 1, 1.0, 0.95, 100.0, 100.0);
    let config = SaemConfig::new()
        .k1_iterations(1)
        .k2_iterations(2)
        .burn_in(0)
        .averaged_iterates(0.75)
        .markov_simulation_variance(markov)
        .covariance_stability(CovarianceStabilityConfig::new(0.99, 1))
        .operational_convergence(oc)
        .compute_map(false)
        .seed(47);
    let mut state = SaemState::from_problem(problem(), &config).unwrap();
    state.step().unwrap();
    state.cycle_diagnostics[0].omega_relative_spd_margin = Some(0.5);
    state.cycle_diagnostics[0].omega_update_rejected = true;

    state.step().unwrap();

    let check = state
        .operational_diagnostics
        .checks
        .last()
        .expect("operational checkpoint");
    let boundary = check
        .criteria
        .iter()
        .find(|criterion| criterion.name == "omega_boundary_rejection_run")
        .expect("Omega boundary criterion");
    assert_eq!(
        boundary.status,
        OperationalConvergenceCriterionStatus::NotSatisfied
    );
    assert_ne!(state.status, Status::Stop(StopReason::Converged));
}

#[test]
fn iov_boundary_rejection_blocks_converged_stop_reason() {
    use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
    let markov = MarkovSimulationVarianceConfig::new(
        7,
        0,
        12,
        6,
        LugsailConfig::over_lugsail_bartlett(),
        4,
        1024 * 1024,
    );
    let config = SaemConfig::new()
        .k1_iterations(1)
        .k2_iterations(2)
        .burn_in(0)
        .averaged_iterates(0.75)
        .markov_simulation_variance(markov)
        .covariance_stability(CovarianceStabilityConfig::new(0.99, 1))
        .operational_convergence(OperationalConvergenceConfig::literature_guided(
            1, 1, 1.0, 0.95, 100.0, 100.0,
        ))
        .compute_map(false)
        .seed(48);
    let mut state = SaemState::from_problem(iov_problem(), &config).unwrap();
    state.step().unwrap();
    state.cycle_diagnostics[0].omega_iov_relative_spd_margin = Some(0.5);
    state.cycle_diagnostics[0].omega_iov_update_rejected = true;
    state.step().unwrap();

    let check = state
        .operational_diagnostics
        .checks
        .last()
        .expect("operational checkpoint");
    let boundary = check
        .criteria
        .iter()
        .find(|criterion| criterion.name == "omega_iov_boundary_rejection_run")
        .expect("Omega_IOV boundary criterion");
    assert_eq!(
        boundary.status,
        OperationalConvergenceCriterionStatus::NotSatisfied
    );
    assert_ne!(state.status, Status::Stop(StopReason::Converged));
}

#[test]
fn operational_convergence_waits_for_complete_covariance_window() {
    use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
    let markov = MarkovSimulationVarianceConfig::new(
        7,
        0,
        12,
        6,
        LugsailConfig::over_lugsail_bartlett(),
        4,
        1024 * 1024,
    );
    let config = SaemConfig::new()
        .k1_iterations(1)
        .k2_iterations(5)
        .burn_in(0)
        .averaged_iterates(0.75)
        .markov_simulation_variance(markov)
        .covariance_stability(CovarianceStabilityConfig::new(0.01, 5))
        .operational_convergence(OperationalConvergenceConfig::literature_guided(
            1, 1, 1.0, 0.95, 100.0, 100.0,
        ))
        .compute_map(false)
        .seed(49);
    let mut state = SaemState::from_problem(problem(), &config).unwrap();
    state.step().unwrap();
    state.step().unwrap();

    let first = state
        .operational_diagnostics
        .checks
        .last()
        .expect("first operational checkpoint");
    let first_boundary = first
        .criteria
        .iter()
        .find(|criterion| criterion.name == "omega_boundary_rejection_run")
        .expect("Omega boundary criterion");
    assert!(matches!(
        first_boundary.status,
        OperationalConvergenceCriterionStatus::Unavailable(_)
    ));
    assert!(matches!(
        first.outcome,
        OperationalConvergenceOutcome::Ineligible { .. }
    ));
    assert_ne!(state.status, Status::Stop(StopReason::Converged));

    while state.cycle < 5 && !state.status.is_stop() {
        state.step().unwrap();
    }
    let eligible = state
        .operational_diagnostics
        .checks
        .last()
        .expect("fifth-cycle operational checkpoint");
    assert_eq!(eligible.iteration, 5);
    let eligible_boundary = eligible
        .criteria
        .iter()
        .find(|criterion| criterion.name == "omega_boundary_rejection_run")
        .expect("Omega boundary criterion");
    assert_eq!(
        eligible_boundary.status,
        OperationalConvergenceCriterionStatus::Satisfied
    );
}

#[test]
fn operational_convergence_final_checkpoint_runs_once_with_truthful_flags() {
    use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
    let markov = MarkovSimulationVarianceConfig::new(
        7,
        0,
        12,
        6,
        LugsailConfig::over_lugsail_bartlett(),
        4,
        1024 * 1024,
    );
    // check_interval=1 means every smoothing iteration is a checkpoint,
    // so the last scheduled checkpoint and the mandatory final will overlap.
    let oc = OperationalConvergenceConfig::literature_guided(1, 1, 1.0, 0.95, 0.1, 0.02);
    let config = SaemConfig::new()
        .k1_iterations(1)
        .k2_iterations(2)
        .burn_in(0)
        .averaged_iterates(0.75)
        .markov_simulation_variance(markov)
        .covariance_stability(CovarianceStabilityConfig::new(0.01, 2))
        .operational_convergence(oc)
        .compute_map(false)
        .seed(45);
    let result = problem().fit_with(config).unwrap();
    let ops = result.operational_diagnostics();
    assert!(!ops.final_check_reused);
    let final_check = ops.checks.last().expect("final checkpoint");
    assert!(final_check.scheduled);
    assert!(final_check.mandatory_final);
    assert_eq!(
        ops.checks
            .iter()
            .filter(|check| check.iteration == final_check.iteration)
            .count(),
        1
    );
}

#[test]
fn operational_convergence_checkpoint_seed_is_deterministic_and_global_seed_is_unchanged() {
    use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
    let markov = MarkovSimulationVarianceConfig::new(
        7,
        0,
        12,
        6,
        LugsailConfig::over_lugsail_bartlett(),
        4,
        1024 * 1024,
    );
    let oc = OperationalConvergenceConfig::literature_guided(1, 1, 1.0, 0.95, 0.1, 0.02);
    let config = SaemConfig::new()
        .k1_iterations(1)
        .k2_iterations(3)
        .burn_in(0)
        .averaged_iterates(0.75)
        .markov_simulation_variance(markov)
        .covariance_stability(CovarianceStabilityConfig::new(0.01, 2))
        .operational_convergence(oc)
        .compute_map(false)
        .seed(46);
    let result1 = problem().fit_with(config.clone()).unwrap();
    let result2 = problem().fit_with(config).unwrap();

    let ops1 = result1.operational_diagnostics();
    let ops2 = result2.operational_diagnostics();
    assert_eq!(ops1.checks.len(), ops2.checks.len());
    for (c1, c2) in ops1.checks.iter().zip(ops2.checks.iter()) {
        assert_eq!(c1.checkpoint_seed, c2.checkpoint_seed);
        assert_eq!(c1.z_quantile, c2.z_quantile);
        assert_eq!(c1.outcome, c2.outcome);
    }
    // Canonical fit result must be unchanged by operational convergence
    assert_eq!(
        result1.population_parameters(),
        result2.population_parameters()
    );
    assert_eq!(result1.omega(), result2.omega());
    assert_eq!(result1.conditional_n2ll(), result2.conditional_n2ll());
}

#[test]
fn normal_two_sided_z_covers_common_confidence_levels() {
    use statrs::distribution::{ContinuousCDF, Normal};
    let norm = Normal::new(0.0, 1.0).unwrap();
    for p in [0.90, 0.95, 0.99] {
        let expected = norm.inverse_cdf(p + (1.0 - p) / 2.0);
        let actual = normal_two_sided_z(p);
        assert!((actual - expected).abs() < 1e-10);
    }
}

#[test]
fn gong_flegal_fixed_width_and_implied_ess_are_exact() {
    let z = normal_two_sided_z(0.95);
    let epsilon = 0.05;
    let implied = 4.0 * z * z / (epsilon * epsilon);
    assert!((implied - 6146.34).abs() < 0.1);
    let boundary_fraction = epsilon / (2.0 * z);
    assert!(2.0 * z * boundary_fraction <= epsilon);
    assert!(2.0 * z * (boundary_fraction + 1e-12) > epsilon);
}

#[test]
fn evaluate_criterion_detects_satisfied_not_satisfied_and_unavailable() {
    let satisfied = evaluate_criterion("test", Some(0.5), 1.0, |v| v <= 1.0);
    assert_eq!(
        satisfied.status,
        OperationalConvergenceCriterionStatus::Satisfied
    );
    assert_eq!(satisfied.observed, Some(0.5));

    let not_satisfied = evaluate_criterion("test", Some(2.0), 1.0, |v| v <= 1.0);
    assert_eq!(
        not_satisfied.status,
        OperationalConvergenceCriterionStatus::NotSatisfied
    );
    assert_eq!(not_satisfied.observed, Some(2.0));

    let unavailable_none = evaluate_criterion("test", None, 1.0, |v| v <= 1.0);
    assert!(matches!(
        unavailable_none.status,
        OperationalConvergenceCriterionStatus::Unavailable(_)
    ));
    assert_eq!(unavailable_none.observed, None);

    let unavailable_nan = evaluate_criterion("test", Some(f64::NAN), 1.0, |v| v <= 1.0);
    assert!(matches!(
        unavailable_nan.status,
        OperationalConvergenceCriterionStatus::Unavailable(_)
    ));
}

#[test]
fn newton_displacement_requires_matching_dimensions() {
    let empty_info = InformationDiagnostics {
        coordinates: vec![],
        recursion_cycles: 0,
        delta: vec![],
        g: vec![],
        expected_complete_hessian: vec![],
        observed_hessian: vec![],
        observed_information: vec![],
        status: InformationStatus::Available,
    };
    let empty_markov = MarkovSimulationVarianceDiagnostics::disabled();
    assert_eq!(newton_displacement(&empty_info, &empty_markov), None);
    assert_eq!(newton_displacement_mc_sd(&empty_info, &empty_markov), None);
}
