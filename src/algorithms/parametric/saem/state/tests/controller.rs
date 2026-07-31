use super::*;
#[test]
fn parametric_fit_controller_steps_like_nonparametric_controller() {
    let config = SaemConfig::new()
        .k1_iterations(2)
        .k2_iterations(1)
        .burn_in(1);
    let mut controller = problem().fit_controller(config).unwrap();

    assert_eq!(controller.cycle(), 0);
    assert!(controller.status().is_continue());
    assert!(controller.likelihood().is_finite());
    assert_eq!(controller.population_parameters(), &[0.2, 10.0]);
    assert_eq!(controller.random_effect_names(), &["ke", "v"]);
    assert_eq!(controller.iov_effect_names(), None);
    assert_eq!(controller.omega_iov(), None);
    assert_eq!(controller.residual_sigmas(), &[0.5]);
    assert_eq!(controller.acceptance_rate(), None);
    assert_eq!(controller.kappa_acceptance_rate(), None);
    assert_eq!(controller.rejected_proposals(), None);
    assert_eq!(controller.non_finite_proposals(), None);
    assert_eq!(controller.parameter_acceptance_rates(), None);
    assert_eq!(
        controller.proposal_step_sizes(),
        Some([0.5, 0.5].as_slice())
    );
    assert!(controller.eta_log_prior().is_finite());
    assert_eq!(
        controller.log_posterior(),
        controller.likelihood() + controller.eta_log_prior()
    );
    assert!(controller.negative_log_likelihood().is_finite());
    assert_eq!(
        controller.negative_log_likelihood(),
        -controller.likelihood()
    );
    assert!(controller.n2ll().is_finite());
    assert_eq!(controller.n_chains(), Some(25));
    assert_eq!(
        controller.omega(),
        Some(&ndarray::array![[1.0, 0.0], [0.0, 1.0]])
    );
    assert_eq!(controller.omega_diagonal(), Some(vec![1.0, 1.0]));
    assert_eq!(
        controller.log_acceptance_ratios(),
        Some([0.0, 0.0].as_slice())
    );
    assert_eq!(controller.total_iterations(), 3);
    assert_eq!(controller.step_size(), 0.0);

    assert!(controller.step().unwrap().is_continue());
    assert_eq!(controller.cycle(), 1);
    assert_eq!(controller.step_size(), 0.0);
    assert_eq!(controller.population_parameters(), &[0.2, 10.0]);
    assert_eq!(
        controller.omega(),
        Some(&ndarray::array![[1.0, 0.0], [0.0, 1.0]])
    );
    assert!(controller.acceptance_rate().is_some());
    assert_eq!(controller.kappa_acceptance_rate(), None);
    assert!(controller.rejected_proposals().is_some());
    assert_eq!(controller.non_finite_proposals(), Some(0));
    let parameter_acceptance_rates = controller.parameter_acceptance_rates().unwrap();
    assert_eq!(parameter_acceptance_rates.len(), 2);
    assert!(parameter_acceptance_rates
        .iter()
        .all(|rate| (0.0..=1.0).contains(rate)));
    assert!(controller.step().unwrap().is_continue());
    assert_eq!(controller.cycle(), 2);
    assert_eq!(controller.step_size(), 1.0);
    assert!(controller.step().unwrap().is_stop());
    assert_eq!(controller.cycle(), 3);
}

#[test]
fn aborted_controller_preserves_typed_termination_reason() {
    let mut controller = problem()
        .fit_controller(SaemConfig::new().compute_map(false))
        .unwrap();
    controller.step().unwrap();
    controller.request_stop();

    let result = controller.into_result().unwrap();

    assert!(!result.converged());
    assert_eq!(result.termination_reason(), Some(&StopReason::Aborted));
    assert_ne!(result.termination_reason(), Some(&StopReason::MaxCycles));
    assert_ne!(
        result.termination_reason(),
        Some(&StopReason::NumericalFailure)
    );
    assert_eq!(result.iterations(), 1);
}

#[test]
fn expectation_numerical_failure_stops_and_blocks_result() {
    let mut state = SaemState::from_problem(
        problem(),
        &SaemConfig::new()
            .n_chains(1)
            .mcmc_iterations(1)
            .compute_map(false),
    )
    .unwrap();
    state.omega[[0, 0]] = f64::NAN;

    let error = state.step().unwrap_err();
    let failure = error
        .downcast_ref::<NumericalFailure>()
        .expect("step error should retain its numerical failure type")
        .clone();

    assert_eq!(failure.attempted_cycle(), 1);
    assert_eq!(failure.phase(), NumericalFailurePhase::Expectation);
    assert!(!failure.source_message().is_empty());
    assert_eq!(state.status, Status::Stop(StopReason::NumericalFailure));
    assert_eq!(
        state.step().unwrap(),
        Status::Stop(StopReason::NumericalFailure)
    );

    let result_error = Box::new(state).into_result().unwrap_err();
    assert_eq!(
        result_error.downcast_ref::<NumericalFailure>(),
        Some(&failure)
    );
}

#[test]
fn maximization_numerical_failure_stops_fit() {
    let mut state = SaemState::from_problem(
        problem(),
        &SaemConfig::new()
            .n_chains(1)
            .mcmc_iterations(1)
            .compute_map(false),
    )
    .unwrap();
    state.sufficient_statistics.mean_phi.pop();

    let error = state.step().unwrap_err();
    let failure = error
        .downcast_ref::<NumericalFailure>()
        .expect("step error should retain its numerical failure type");

    assert_eq!(failure.attempted_cycle(), 1);
    assert_eq!(failure.phase(), NumericalFailurePhase::Maximization);
    assert!(!failure.source_message().is_empty());
    assert_eq!(state.status, Status::Stop(StopReason::NumericalFailure));
}

#[test]
fn result_assembly_numerical_failure_returns_no_result() {
    let mut state =
        SaemState::from_problem(problem(), &SaemConfig::new().n_chains(1).compute_map(false))
            .unwrap();
    state.etas[0].clear();

    let error = Box::new(state).into_result().unwrap_err();
    let failure = error
        .downcast_ref::<NumericalFailure>()
        .expect("result error should retain its numerical failure type");

    assert_eq!(failure.attempted_cycle(), 0);
    assert_eq!(failure.phase(), NumericalFailurePhase::ResultAssembly);
    assert!(!failure.source_message().is_empty());
}

#[test]
fn proposal_score_uses_pmcore_likelihood_and_eta_prior() {
    let state = SaemState::from_problem(problem(), &SaemConfig::new().n_chains(1)).unwrap();
    let current_eta = state.etas[0][0].clone();
    let score = state
        .score_subject_latents(0, &current_eta, &state.kappas[0][0])
        .unwrap();

    assert_eq!(score.log_likelihood, state.subject_log_likelihoods[0]);
    assert_eq!(score.eta_log_prior, state.subject_log_priors[0]);
    assert_eq!(
        state
            .proposal_log_acceptance_ratio(0, 0, &current_eta)
            .unwrap(),
        0.0
    );
}

#[test]
fn component_random_walk_changes_only_selected_eta() {
    let mut state =
        SaemState::from_problem(problem(), &SaemConfig::new().n_chains(2).seed(2024)).unwrap();
    let current = vec![1.0, 2.0];

    let proposed = state.component_random_walk_eta(&current, 1);

    assert_eq!(proposed[0], current[0]);
    assert_ne!(proposed[1], current[1]);
}

#[test]
fn component_scale_adaptation_uses_acceptance_bands_and_clamps() {
    assert!((adapt_component_step_size(1.0, 0.45) - 1.1).abs() < 1e-12);
    assert!((adapt_component_step_size(1.0, 0.44) - 0.9).abs() < 1e-12);
    assert_eq!(adapt_component_step_size(5.0, 1.0), 5.0);
    assert_eq!(adapt_component_step_size(1e-6, 0.0), 1e-6);
}

#[test]
fn saemix_acceptance_and_adaptation_formulas_are_explicit() {
    let current = SubjectPosteriorScore {
        log_likelihood: -8.0,
        eta_log_prior: -1.0,
        kappa_log_prior: 0.0,
    };
    let proposed = SubjectPosteriorScore {
        log_likelihood: -6.5,
        eta_log_prior: -20.0,
        kappa_log_prior: 0.0,
    };
    assert_eq!(
        saemix_prior_independence_log_acceptance(current, proposed),
        1.5
    );

    let policy = SaemixMcmcConfig::new([0, 1, 0, 0]);
    assert!((saemix_adapt_step_size(0.5, 0.5, policy).unwrap() - 0.52).abs() < 1e-12);
    assert!((saemix_adapt_step_size(0.5, 0.3, policy).unwrap() - 0.48).abs() < 1e-12);

    let covariance = ndarray::array![[1.0]];
    let ratio =
        saemix_map_independence_log_acceptance(current, proposed, &[0.5], &[1.0], &covariance)
            .unwrap();
    let expected = proposed.log_posterior() - current.log_posterior()
        + eta_log_prior_from_omega(&[0.5], &covariance).unwrap()
        - eta_log_prior_from_omega(&[1.0], &covariance).unwrap();
    assert_eq!(ratio, expected);
}

#[test]
fn saemix_rotating_subset_schedule_changes_only_selected_coordinates() {
    let mut state =
        SaemState::from_problem(problem(), &SaemConfig::new().n_chains(1).seed(2024)).unwrap();
    state.cycle = 2;
    let (subset_size, groups) = state.saemix_subset_groups(3);
    assert_eq!(subset_size, 2);
    assert_eq!(groups.len(), 3);
    assert!(groups.iter().all(|group| group.len() == 2));
    assert!(groups
        .iter()
        .all(|group| group[0] != group[1] && group.iter().all(|index| *index < 3)));

    state.saemix_subset_step_sizes = vec![vec![0.5; 3]; 3];
    let current = vec![1.0, 2.0, 3.0];
    let proposed = state.subset_random_walk_eta(&current, &groups[0], subset_size);
    for index in 0..3 {
        if groups[0].contains(&index) {
            assert_ne!(proposed[index], current[index]);
        } else {
            assert_eq!(proposed[index], current[index]);
        }
    }
}

#[test]
fn saemix_four_kernel_policy_records_order_counts_and_map_window() {
    let config = SaemConfig::new()
        .n_chains(1)
        .k1_iterations(2)
        .k2_iterations(0)
        .burn_in(2)
        .compute_map(false)
        .saemix_mcmc_config(SaemixMcmcConfig::new([1, 1, 1, 1]).map_cycles(2));
    let mut state = SaemState::from_problem(problem(), &config).unwrap();

    state.step().unwrap();
    state.step().unwrap();

    let expected = [
        SaemMcmcKernel::PriorIndependence,
        SaemMcmcKernel::ComponentRandomWalk,
        SaemMcmcKernel::RotatingSubset,
        SaemMcmcKernel::MapIndependence,
    ];
    for cycle in &state.cycle_diagnostics {
        assert_eq!(
            cycle
                .mcmc_kernel_diagnostics
                .iter()
                .map(|diagnostic| diagnostic.kernel)
                .collect::<Vec<_>>(),
            expected
        );
        assert!(cycle
            .mcmc_kernel_diagnostics
            .iter()
            .all(|diagnostic| diagnostic.accepted + diagnostic.rejected == diagnostic.proposals));
        assert!(cycle
            .mcmc_kernel_diagnostics
            .iter()
            .all(|diagnostic| diagnostic.non_finite == 0));
    }
    assert!(state.cycle_diagnostics[0].mcmc_kernel_diagnostics[0].proposals > 0);
    assert!(state.cycle_diagnostics[0].mcmc_kernel_diagnostics[1].proposals > 0);
    assert!(state.cycle_diagnostics[0].mcmc_kernel_diagnostics[2].proposals > 0);
    assert!(state.cycle_diagnostics[0].mcmc_kernel_diagnostics[3].proposals > 0);
    assert_eq!(
        state.cycle_diagnostics[1].mcmc_kernel_diagnostics[3].proposals,
        0
    );
}

#[test]
fn saemix_compatibility_fails_closed_for_iov() {
    let error =
        SaemState::from_problem(iov_problem(), &SaemConfig::new().saemix_mcmc([0, 1, 0, 0]))
            .expect_err("SAEMix compatibility must not silently approximate IOV")
            .to_string();
    assert!(error.contains("does not support IOV"));
}

#[test]
fn component_scale_adaptation_waits_for_interval_and_resets_counts() {
    let mut state =
        SaemState::from_problem(problem(), &SaemConfig::new().n_chains(2).adapt_interval(2))
            .unwrap();
    state.adaptation_accept_counts = vec![9, 1];
    state.adaptation_proposal_counts = vec![10, 10];
    state.steps_since_adapt = 1;

    state.adapt_proposal_step_sizes();
    assert_eq!(state.proposal_step_sizes, vec![0.5, 0.5]);

    state.steps_since_adapt = 2;
    state.adapt_proposal_step_sizes();
    assert_eq!(state.proposal_step_sizes, vec![0.55, 0.45]);
    assert_eq!(state.adaptation_accept_counts, vec![0, 0]);
    assert_eq!(state.adaptation_proposal_counts, vec![0, 0]);
    assert_eq!(state.steps_since_adapt, 0);
}

#[test]
fn e_step_runs_seeded_random_walk_for_all_chains_and_records_acceptance_rate() {
    let config = SaemConfig::new().n_chains(3).mcmc_iterations(2).seed(2024);
    let mut state = SaemState::from_problem(problem(), &config).unwrap();
    let initial_etas = state.etas.clone();

    state.e_step().unwrap();

    let acceptance_rate = state.acceptance_rate().unwrap();
    assert!((0.0..=1.0).contains(&acceptance_rate));
    assert_eq!(state.last_log_acceptance_ratios.len(), 2);
    assert_eq!(state.last_parameter_acceptance_rates.len(), 2);
    assert!(state
        .last_parameter_acceptance_rates
        .iter()
        .all(|rate| (0.0..=1.0).contains(rate)));
    assert!(state
        .last_log_acceptance_ratios
        .iter()
        .all(|value| value.is_finite()));
    assert_ne!(state.etas, initial_etas);
    assert!(state
        .etas
        .iter()
        .flat_map(|subject_chains| subject_chains.iter())
        .all(|eta| eta.len() == 2));
}

#[test]
fn cycle_diagnostics_separate_eta_kappa_counts_and_schedule_phases() {
    let config = SaemConfig::new()
        .n_chains(2)
        .mcmc_iterations(1)
        .burn_in(1)
        .k1_iterations(2)
        .k2_iterations(1);
    let mut state = SaemState::from_problem(iov_problem(), &config).unwrap();

    state.step().unwrap();
    state.step().unwrap();
    state.step().unwrap();

    assert_eq!(state.cycle_diagnostics.len(), 3);
    assert_eq!(state.cycle_diagnostics[0].phase, SaemPhase::BurnIn);
    assert_eq!(state.cycle_diagnostics[1].phase, SaemPhase::Exploration);
    assert_eq!(state.cycle_diagnostics[2].phase, SaemPhase::Smoothing);
    for diagnostics in &state.cycle_diagnostics {
        assert_eq!(diagnostics.eta_proposals, 4);
        assert_eq!(
            diagnostics.eta_accepted + diagnostics.eta_rejected,
            diagnostics.eta_proposals
        );
        assert_eq!(diagnostics.kappa_proposals, 4);
        assert_eq!(
            diagnostics.kappa_accepted + diagnostics.kappa_rejected,
            diagnostics.kappa_proposals
        );
        assert_eq!(diagnostics.eta_parameter_acceptance_rates.len(), 2);
        assert_eq!(
            diagnostics.eta_proposal_step_sizes_before_adaptation.len(),
            2
        );
        assert_eq!(
            diagnostics.eta_proposal_step_sizes_after_adaptation.len(),
            2
        );
        assert_eq!(diagnostics.kappa_subject_acceptance_rates.len(), 1);
        assert_eq!(
            diagnostics
                .kappa_proposal_step_sizes_before_adaptation
                .len(),
            1
        );
        assert_eq!(
            diagnostics.kappa_proposal_step_sizes_after_adaptation.len(),
            1
        );
    }
    assert_eq!(
        state.cycle_diagnostics[0].stochastic_approximation_step,
        0.0
    );
    assert_eq!(state.cycle_diagnostics[0].covariance_step, 0.1);
}

#[test]
fn warning_aggregation_preserves_kind_output_first_cycle_and_counts() {
    let config = SaemConfig::new()
        .n_chains(1)
        .mcmc_iterations(1)
        .burn_in(0)
        .k1_iterations(1)
        .k2_iterations(0);
    let mut state = SaemState::from_problem(problem(), &config).unwrap();
    state.step().unwrap();
    let cycle = &mut state.cycle_diagnostics[0];
    cycle.omega_update_rejected = true;
    cycle.eta_non_finite = 2;
    cycle.eta_block_non_finite = 7;
    let residual = &mut cycle.residual_diagnostics[0];
    residual.update_rejected = true;
    residual.proportional_floor_count = 3;
    residual.non_finite_prediction_count = 4;
    residual.exponential_domain_violation_count = 5;
    residual.combined_additive_collapse_warning = true;
    residual.optimizer_converged = Some(false);

    let warnings = parametric_warnings(&state.cycle_diagnostics, None);

    assert!(warnings.contains(&ParametricWarning::OmegaUpdateRejected {
        first_iteration: 1,
        cycles: 1,
    }));
    assert!(
        warnings.contains(&ParametricWarning::EtaNonFiniteProposals {
            first_iteration: 1,
            count: 2,
        })
    );
    assert!(
        warnings.contains(&ParametricWarning::EtaBlockNonFiniteProposals {
            first_iteration: 1,
            count: 7,
        })
    );
    assert!(
        warnings.contains(&ParametricWarning::ResidualUpdateRejected {
            output: "0".to_owned(),
            first_iteration: 1,
            cycles: 1,
        })
    );
    assert!(
        warnings.contains(&ParametricWarning::ProportionalPredictionFloor {
            output: "0".to_owned(),
            first_iteration: 1,
            count: 3,
        })
    );
    assert!(
        warnings.contains(&ParametricWarning::NonFiniteResidualPrediction {
            output: "0".to_owned(),
            first_iteration: 1,
            count: 4,
        })
    );
    assert!(
        warnings.contains(&ParametricWarning::ExponentialDomainViolation {
            output: "0".to_owned(),
            first_iteration: 1,
            count: 5,
        })
    );
    assert!(
        warnings.contains(&ParametricWarning::CombinedAdditiveCollapse {
            output: "0".to_owned(),
            first_iteration: 1,
            cycles: 1,
        })
    );
    assert!(
        warnings.contains(&ParametricWarning::ResidualOptimizerNotConverged {
            output: "0".to_owned(),
            first_iteration: 1,
            cycles: 1,
        })
    );
}

#[test]
fn covariance_stability_records_fixed_iiv_and_iov_margins_and_output_rows() {
    let result = markov_iov_problem()
        .fit_with(
            SaemConfig::new()
                .n_chains(1)
                .mcmc_iterations(1)
                .burn_in(0)
                .k1_iterations(1)
                .k2_iterations(0)
                .covariance_stability(CovarianceStabilityConfig::new(0.01, 1)),
        )
        .unwrap();
    let cycle = &result.cycle_diagnostics()[0];
    assert!((cycle.omega_relative_spd_margin.unwrap() - 1.0).abs() < 1e-12);
    assert!((cycle.omega_iov_relative_spd_margin.unwrap() - 1.0).abs() < 1e-12);

    let tables = result.tables(0.0, 0.0).unwrap();
    let stability_rows = tables
        .statistics
        .iter()
        .filter(|row| row.kind == "covariance_stability")
        .collect::<Vec<_>>();
    assert_eq!(stability_rows.len(), 2);
    assert!(stability_rows.iter().any(|row| {
        row.name == "omega_relative_spd_margin"
            && row.value.is_some_and(|value| (value - 1.0).abs() < 1e-12)
    }));
    assert!(stability_rows.iter().any(|row| {
        row.name == "omega_iov_relative_spd_margin"
            && row.value.is_some_and(|value| (value - 1.0).abs() < 1e-12)
    }));
}

#[test]
fn covariance_boundary_rejection_requires_a_complete_consecutive_window() {
    let config = SaemConfig::new()
        .n_chains(1)
        .mcmc_iterations(1)
        .burn_in(0)
        .k1_iterations(1)
        .k2_iterations(0);
    let mut state = SaemState::from_problem(problem(), &config).unwrap();
    state.step().unwrap();
    let base = state.cycle_diagnostics[0].clone();
    let policy = CovarianceStabilityConfig::new(0.01, 3);
    let pattern = [
        (1, 0.005, true),
        (2, 0.004, true),
        (3, 0.02, true),
        (4, 0.003, true),
        (5, 0.002, true),
        (6, 0.001, true),
    ];
    let cycles = pattern
        .into_iter()
        .map(|(iteration, margin, rejected)| {
            let mut cycle = base.clone();
            cycle.iteration = iteration;
            cycle.omega_relative_spd_margin = Some(margin);
            cycle.omega_update_rejected = rejected;
            cycle
        })
        .collect::<Vec<_>>();

    assert_eq!(
        covariance_boundary_rejection_summary(&cycles[..2], policy, false),
        CovarianceBoundaryRejectionSummary {
            first_iteration: None,
            longest_run: 2,
        }
    );
    assert_eq!(
        covariance_boundary_rejection_summary(&cycles, policy, false),
        CovarianceBoundaryRejectionSummary {
            first_iteration: Some(4),
            longest_run: 3,
        }
    );
    let warnings = parametric_warnings(&cycles, Some(policy));
    assert!(
        warnings.contains(&ParametricWarning::OmegaBoundaryRejection {
            first_iteration: 4,
            longest_run: 3,
        })
    );

    let mut mismatched_iov = base.clone();
    mismatched_iov.omega_iov_relative_spd_margin = Some(0.005);
    mismatched_iov.omega_update_rejected = true;
    mismatched_iov.omega_iov_update_rejected = false;
    assert_eq!(
        covariance_boundary_rejection_summary(&[mismatched_iov.clone()], policy, true),
        CovarianceBoundaryRejectionSummary::default()
    );
    mismatched_iov.omega_iov_update_rejected = true;
    assert_eq!(
        covariance_boundary_rejection_summary(&[mismatched_iov], policy, true).longest_run,
        1
    );

    let iov_cycles = (1..=3)
        .map(|iteration| {
            let mut cycle = base.clone();
            cycle.iteration = iteration;
            cycle.omega_iov_relative_spd_margin = Some(policy.minimum_relative_spd_margin);
            cycle.omega_iov_update_rejected = true;
            cycle
        })
        .collect::<Vec<_>>();
    assert_eq!(
        covariance_boundary_rejection_summary(&iov_cycles, policy, true),
        CovarianceBoundaryRejectionSummary {
            first_iteration: Some(1),
            longest_run: 3,
        }
    );
    assert!(parametric_warnings(&iov_cycles, Some(policy)).contains(
        &ParametricWarning::OmegaIovBoundaryRejection {
            first_iteration: 1,
            longest_run: 3,
        }
    ));

    let criterion = evaluate_criterion(
        "omega_boundary_rejection_run",
        Some(3.0),
        policy.rejection_window as f64,
        |observed| observed < policy.rejection_window as f64,
    );
    assert_eq!(
        criterion.status,
        OperationalConvergenceCriterionStatus::NotSatisfied
    );
}

#[test]
fn m_step_recenters_etas_before_updating_iiv_second_moment() {
    let mut state = SaemState::from_problem(
        problem(),
        &SaemConfig::new()
            .n_chains(1)
            .burn_in(0)
            .omega_sa_max_step(0.1),
    )
    .unwrap();
    state.cycle = 1;
    for subject_chains in &mut state.etas {
        for eta in subject_chains {
            eta[0] = 2.0_f64.ln();
        }
    }
    let individual_before = state.individual_parameters(0, 0);

    state.m_step().unwrap();

    let individual_after = state.individual_parameters(0, 0);
    assert!((individual_before[0] - individual_after[0]).abs() < 1e-12);
    assert!(state
        .etas
        .iter()
        .flat_map(|subject_chains| subject_chains.iter())
        .all(|eta| eta[0].abs() < 1e-12));
    assert!((state.population_parameters[0] - 0.4).abs() < 1e-12);
    assert!((state.population_parameters[1] - 10.0).abs() < 1e-12);
    let information = state.information.diagnostics();
    let ke_coordinate = information
        .coordinates
        .iter()
        .position(|coordinate| coordinate.name == "phi:ke")
        .unwrap();
    // Two pre-M-step absolute phi values each differ from the old
    // population by ln(2). Post-update or un-recentered evaluation would
    // give a different score (zero or double-counted population shift).
    assert!((information.delta[ke_coordinate] - 2.0 * 2.0_f64.ln()).abs() < 1e-12);
    let expected_omega = ndarray::array![[0.9, 0.0], [0.0, 0.9]];
    assert!(state
        .iiv_second_moment
        .iter()
        .zip(expected_omega.iter())
        .all(|(actual, expected)| (actual - expected).abs() < 1e-12));
    assert!(state
        .omega
        .iter()
        .zip(expected_omega.iter())
        .all(|(actual, expected)| (actual - expected).abs() < 1e-12));
}
