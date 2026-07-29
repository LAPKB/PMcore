use super::*;
#[test]
fn explicit_terminal_policy_preserves_default_trajectory() {
    let base = SaemConfig::new()
        .k1_iterations(2)
        .burn_in(1)
        .k2_iterations(2)
        .compute_map(false)
        .seed(7788);
    let default = problem().fit_with(base.clone()).unwrap();
    let explicit = problem()
        .fit_with(base.estimator_policy(SaemEstimatorPolicy::TerminalIterate))
        .unwrap();
    assert_eq!(default.cycle_diagnostics(), explicit.cycle_diagnostics());
    assert_eq!(
        default.population_parameters(),
        explicit.population_parameters()
    );
    assert_eq!(default.omega(), explicit.omega());
    assert_eq!(default.conditional_n2ll(), explicit.conditional_n2ll());
    assert_eq!(default.termination_reason(), Some(&StopReason::MaxCycles));
    assert_eq!(explicit.termination_reason(), Some(&StopReason::MaxCycles));
}

fn residual_phase_schedule() -> SaemSchedule {
    let mut schedule = SaemSchedule::from_config(
        &SaemConfig::new()
            .burn_in(0)
            .k1_iterations(4)
            .k2_iterations(3),
    );
    schedule.variance_floor_iterations = 1;
    schedule
}

#[test]
fn combined_residual_component_anneals_during_configured_period() {
    let schedule = residual_phase_schedule();
    let applied = applied_combined_residual_component(&schedule, 1, 1.0, 0.1, true);
    assert_eq!(applied, schedule.annealing_alpha);
}

#[test]
fn combined_residual_component_replaces_directly_in_remaining_exploration() {
    let schedule = residual_phase_schedule();
    assert_eq!(
        applied_combined_residual_component(&schedule, 2, 1.0, 0.1, true),
        0.1
    );
}

#[test]
fn combined_residual_component_smooths_in_k2() {
    let schedule = residual_phase_schedule();
    assert_eq!(
        applied_combined_residual_component(&schedule, 6, 1.0, 0.2, true),
        0.6
    );
}

#[test]
fn combined_residual_component_preserves_fixed_value() {
    let schedule = residual_phase_schedule();
    assert_eq!(
        applied_combined_residual_component(&schedule, 1, 1.0, 0.1, false),
        1.0
    );
    assert_eq!(
        applied_combined_residual_component(&schedule, 6, 1.0, 0.1, false),
        1.0
    );
}

#[test]
fn burn_in_warms_covariance_statistics_without_updating_parameters() {
    let config = SaemConfig::new()
        .n_chains(1)
        .burn_in(2)
        .k1_iterations(4)
        .omega_sa_max_step(0.1);
    let mut state = SaemState::from_problem(problem(), &config).unwrap();
    for subject_chains in &mut state.etas {
        subject_chains[0].fill(2.0);
    }
    let initial_population = state.population_parameters.clone();
    let initial_omega = state.omega.clone();
    let initial_iiv_second_moment = state.iiv_second_moment.clone();
    let initial_phi_second_moment = state.sufficient_statistics.second_moment.clone();

    state.step().unwrap();

    assert_eq!(state.cycle, 1);
    assert_eq!(state.cycle_diagnostics[0].phase, SaemPhase::BurnIn);
    assert_eq!(state.population_parameters, initial_population);
    assert_eq!(state.omega, initial_omega);
    assert_ne!(state.iiv_second_moment, initial_iiv_second_moment);
    assert_ne!(
        state.sufficient_statistics.second_moment,
        initial_phi_second_moment
    );
}

#[test]
fn chain_count_auto_scales_for_small_datasets() {
    assert_eq!(n_chains(&SaemConfig::default(), 2), 25);
    assert_eq!(n_chains(&SaemConfig::new().n_chains(3), 2), 3);
    assert_eq!(n_chains(&SaemConfig::default(), 100), 1);
}

#[test]
fn result_retains_requested_config_and_separate_effective_chain_count() {
    let config = SaemConfig::new()
        .n_chains(1)
        .k1_iterations(1)
        .k2_iterations(0)
        .burn_in(1)
        .compute_map(false)
        .seed(9876);
    let serialized_config = serde_json::to_value(&config).unwrap();
    let state = SaemState::from_problem(problem(), &config).unwrap();

    let result = Box::new(state).into_result().unwrap();

    assert_eq!(result.config().n_chains, 1);
    assert_eq!(result.effective_n_chains(), 25);
    assert_eq!(
        serde_json::to_value(result.config()).unwrap(),
        serialized_config
    );
}

#[test]
fn result_parameter_metadata_preserves_declaration_order() {
    let config = SaemConfig::new()
        .k1_iterations(1)
        .k2_iterations(0)
        .burn_in(1)
        .compute_map(false);
    let state = SaemState::from_problem(ordered_metadata_problem(), &config).unwrap();

    let result = Box::new(state).into_result().unwrap();

    assert_eq!(result.parameter_names(), ["ke", "v"]);
    assert_eq!(
        result.parameter_scales(),
        [ParameterScale::Identity, ParameterScale::Log]
    );
    assert_eq!(result.estimated_parameters(), [true, false]);
    assert_eq!(result.random_effect_indices(), [0]);
    assert_eq!(result.random_effect_names(), ["ke"]);
    assert_eq!(result.iov_effect_indices(), [1]);
    assert_eq!(result.iov_effect_names(), ["v"]);
}

#[test]
fn result_retains_exact_symmetric_iiv_covariance_masks() {
    let config = SaemConfig::new()
        .k1_iterations(1)
        .k2_iterations(0)
        .burn_in(1)
        .compute_map(false);
    let configured =
        Box::new(SaemState::from_problem(configured_omega_problem(), &config).unwrap())
            .into_result()
            .unwrap();
    let correlated =
        Box::new(SaemState::from_problem(correlated_omega_problem(), &config).unwrap())
            .into_result()
            .unwrap();

    assert_eq!(configured.random_effect_names(), ["ke", "v"]);
    assert_eq!(
        configured.omega_structural_mask(),
        &ndarray::array![[true, false], [false, true]]
    );
    assert_eq!(
        configured.omega_estimated_mask(),
        &ndarray::array![[true, false], [false, false]]
    );
    assert_eq!(
        correlated.omega_structural_mask(),
        &ndarray::array![[true, true], [true, true]]
    );
    assert_eq!(
        correlated.omega_estimated_mask(),
        &ndarray::array![[true, true], [true, true]]
    );
}

#[test]
fn result_retains_ordered_iov_masks_and_none_without_iov() {
    let config = SaemConfig::new()
        .k1_iterations(1)
        .k2_iterations(0)
        .burn_in(1)
        .compute_map(false);
    let iov = Box::new(SaemState::from_problem(configured_iov_problem(), &config).unwrap())
        .into_result()
        .unwrap();
    let no_iov = Box::new(SaemState::from_problem(problem(), &config).unwrap())
        .into_result()
        .unwrap();

    assert_eq!(iov.iov_effect_indices(), [0, 1]);
    assert_eq!(iov.iov_effect_names(), ["ke", "v"]);
    assert_eq!(
        iov.omega_iov_structural_mask(),
        Some(&ndarray::array![[true, true], [true, true]])
    );
    assert_eq!(
        iov.omega_iov_estimated_mask(),
        Some(&ndarray::array![[true, false], [false, false]])
    );
    assert_eq!(no_iov.omega_iov_structural_mask(), None);
    assert_eq!(no_iov.omega_iov_estimated_mask(), None);
}

#[test]
fn state_initializes_zero_eta_chains() {
    let state = SaemState::from_problem(problem(), &SaemConfig::default()).unwrap();

    assert_eq!(state.etas.len(), 2);
    assert_eq!(state.etas[0].len(), 25);
    assert_eq!(state.etas[0][0], vec![0.0, 0.0]);
    assert_eq!(state.etas[1][24], vec![0.0, 0.0]);
    assert_eq!(state.omega_diagonal(), Some(vec![1.0, 1.0]));
}

#[test]
fn covariate_state_joint_gls_rebases_eta_and_builds_subject_omega() {
    let mut state = SaemState::from_problem(
        covariate_problem(),
        &SaemConfig::new().n_chains(2).compute_map(false),
    )
    .unwrap();
    let intercept = [0.2_f64.ln(), 10.0_f64.ln()];
    let beta = 0.35;
    let expected_phi = [-1.0, 0.0, 1.0]
        .into_iter()
        .map(|design| vec![intercept[0] + beta * design, intercept[1]])
        .collect::<Vec<_>>();
    let desired_omega = ndarray::array![[0.4, 0.1], [0.1, 0.3]];
    let mut second = desired_omega.clone();
    for mean in &expected_phi {
        for row in 0..2 {
            for column in 0..2 {
                second[[row, column]] += mean[row] * mean[column] / 3.0;
            }
        }
    }
    let old_means = state.subject_mu_phi.clone().unwrap();
    for chains in &mut state.etas {
        for eta in chains {
            eta[0] = 0.1;
            eta[1] = -0.2;
        }
    }
    let absolute_before = old_means
        .iter()
        .map(|mean| vec![mean[0] + 0.1, mean[1] - 0.2])
        .collect::<Vec<_>>();
    state.covariate_statistics = Some(CovariateSufficientStatistics {
        expected_phi,
        global_second_moment: second,
    });

    let candidate = state
        .update_covariate_population_and_recenter_etas()
        .unwrap();
    let model = state.covariate_model.as_ref().unwrap();
    assert!((model.estimates()[0].estimate() - beta).abs() < 1e-10);
    assert!((candidate[[0, 0]] - desired_omega[[0, 0]]).abs() < 1e-10);
    assert!((candidate[[0, 1]] - desired_omega[[0, 1]]).abs() < 1e-10);
    for (subject, mean) in state.subject_mu_phi.as_ref().unwrap().iter().enumerate() {
        for coordinate in 0..2 {
            assert!(
                (mean[coordinate] + state.etas[subject][0][coordinate]
                    - absolute_before[subject][coordinate])
                    .abs()
                    < 1e-10
            );
        }
    }
}

#[test]
fn covariate_fit_executes_and_retains_subject_population_parameters() {
    let result = covariate_problem()
        .fit_with(
            SaemConfig::new()
                .n_chains(2)
                .mcmc_iterations(1)
                .burn_in(1)
                .k1_iterations(2)
                .k2_iterations(2)
                .averaged_iterates(0.75)
                .compute_map(false),
        )
        .unwrap();
    assert!(result.estimator_metadata().average_applied);
    assert_eq!(result.covariate_estimates().unwrap().len(), 2);
    assert!(result.covariate_estimates().unwrap()[0].estimate() < 0.0);
    assert_eq!(
        result
            .covariate_subject_population_parameters()
            .unwrap()
            .unwrap()
            .len(),
        3
    );
    assert!(result.cycle_diagnostics().iter().all(|cycle| cycle
        .covariate_betas
        .as_ref()
        .is_some_and(|values| values.len() == 2)));
    let tables = result.tables(1.0, 0.0).unwrap();
    assert_eq!(tables.covariate_effects.len(), 2);
    assert_eq!(tables.subject_covariates.len(), 6);
    assert_eq!(tables.subject_population_parameters.len(), 6);

    let directory =
        std::env::temp_dir().join(format!("pmcore-schema7-covariate-{}", std::process::id()));
    result.write_outputs(&directory, 1.0, 0.0).unwrap();
    let record =
        crate::results::ParametricResultRecord::read_json(directory.join("result.json")).unwrap();
    assert_eq!(record.schema_version, 9);
    assert_eq!(record.source_metadata.covariate_effects.len(), 2);
    let warm = record
        .warm_start_problem(one_compartment(), result.data().clone())
        .unwrap();
    let warm_estimates = warm
        .covariates()
        .unwrap()
        .estimates()
        .iter()
        .map(|estimate| estimate.estimate())
        .collect::<Vec<_>>();
    let result_estimates = result
        .covariate_estimates()
        .unwrap()
        .iter()
        .map(|estimate| estimate.estimate())
        .collect::<Vec<_>>();
    assert!(warm_estimates
        .iter()
        .zip(result_estimates)
        .all(|(warm, result)| (warm - result).abs() <= 2.0 * f64::EPSILON));
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn fixed_covariate_without_iiv_executes_subject_specific_predictions() {
    let result = fixed_covariate_without_iiv_problem()
        .fit_with(
            SaemConfig::new()
                .n_chains(1)
                .mcmc_iterations(1)
                .burn_in(0)
                .k1_iterations(1)
                .k2_iterations(0)
                .compute_map(false),
        )
        .unwrap();
    assert!(result.random_effect_names().is_empty());
    let means = result
        .covariate_subject_population_parameters()
        .unwrap()
        .unwrap();
    assert!((means[0].psi()[0] - 0.2).abs() < 1e-12);
    assert!((means[1].psi()[0] - 0.2 * 0.2_f64.exp()).abs() < 1e-12);
    let predictions = result.population_predictions(0.0, 0.0).unwrap();
    assert_ne!(
        predictions[0].predictions()[0].prediction(),
        predictions[1].predictions()[0].prediction()
    );
}

#[test]
fn explicit_iiv_mask_controls_eta_and_omega_dimensions() {
    let mut state =
        SaemState::from_problem(partial_iiv_problem(), &SaemConfig::new().n_chains(2)).unwrap();

    assert_eq!(state.initialization.random_effect_indices, vec![0]);
    assert_eq!(state.initialization.random_effect_names, vec!["ke"]);
    assert!(state
        .etas
        .iter()
        .flat_map(|subject_chains| subject_chains.iter())
        .all(|eta| eta.len() == 1));
    assert_eq!(state.omega.dim(), (1, 1));
    assert_eq!(state.proposal_step_sizes.len(), 1);

    state.etas[0][0][0] = 2.0_f64.ln();
    let individual = state.individual_parameters(0, 0);
    assert!((individual[0] - 0.4).abs() < 1e-12);
    assert!((individual[1] - 10.0).abs() < 1e-12);
}

#[test]
fn all_fixed_parameters_support_zero_dimensional_iiv() {
    let config = SaemConfig::new()
        .n_chains(1)
        .burn_in(1)
        .k1_iterations(1)
        .k2_iterations(1);
    let state = SaemState::from_problem(fixed_no_iiv_problem(), &config).expect(
        "fixed population plus estimated residual error should support zero-dimensional IIV",
    );
    assert!(state.initialization.random_effect_names.is_empty());
    assert!(state.omega.is_empty());
    assert!(state.iiv_second_moment.is_empty());
    assert!(state
        .etas
        .iter()
        .all(|chains| chains.iter().all(Vec::is_empty)));

    let result = fixed_no_iiv_problem().fit_with(config).unwrap();
    assert_eq!(result.termination_reason(), Some(&StopReason::MaxCycles));
    assert_eq!(result.iterations(), 2);
    assert!(result.objf().is_finite());
    assert!(result.conditional_modes().is_empty());
    assert_eq!(result.omega_structural_mask().dim(), (0, 0));
    assert_eq!(result.omega_estimated_mask().dim(), (0, 0));
    assert!(result.omega_structural_mask().is_empty());
    assert!(result.omega_estimated_mask().is_empty());
    assert_eq!(result.omega_iov_structural_mask(), None);
    assert_eq!(result.omega_iov_estimated_mask(), None);
    assert!(result
        .eta_chain_means()
        .iter()
        .all(|estimate| estimate.values.is_empty()));
    assert!(result.kappa_chain_means().is_empty());
}

#[test]
fn iov_state_tracks_one_kappa_per_subject_occasion_and_chain() {
    let state = SaemState::from_problem(iov_problem(), &SaemConfig::new().n_chains(2)).unwrap();

    assert_eq!(state.initialization.iov_effect_names, vec!["ke"]);
    assert_eq!(state.omega_iov, Some(ndarray::array![[0.1]]));
    assert_eq!(state.kappas.len(), 1);
    assert_eq!(state.kappas[0].len(), 2);
    assert_eq!(state.kappas[0][0], vec![vec![0.0], vec![0.0]]);
}

#[test]
fn uneven_occasion_counts_preserve_kappa_shapes_order_and_named_lookup() {
    let result = uneven_iov_problem()
        .fit_with(
            SaemConfig::new()
                .n_chains(2)
                .mcmc_iterations(1)
                .burn_in(0)
                .k1_iterations(2)
                .k2_iterations(0)
                .compute_map(false),
        )
        .unwrap();

    assert_eq!(result.kappa_chain_means().len(), 6);
    assert!(result.kappa_chain_mean("one", 0).is_some());
    assert!(result.kappa_chain_mean("one", 1).is_none());
    assert!(result.kappa_chain_mean("two", 0).is_some());
    assert!(result.kappa_chain_mean("two", 1).is_some());
    assert!(result.kappa_chain_mean("three", 0).is_some());
    assert!(result.kappa_chain_mean("three", 1).is_some());
    assert!(result.kappa_chain_mean("three", 2).is_some());
    assert!(result.eta_chain_mean("two").is_some());
    assert!(result.eta_chain_mean("missing").is_none());
    assert!(result.conditional_mode("two").is_none());
    assert!(result
        .cycle_diagnostics()
        .iter()
        .all(|cycle| cycle.kappa_proposals == 12));
}

#[test]
fn iov_scores_per_occasion_kappa_prior_and_conditional_proposal() {
    let state = SaemState::from_problem(iov_problem(), &SaemConfig::new().n_chains(2)).unwrap();
    let score = state
        .score_subject_latents(0, &state.etas[0][0], &state.kappas[0][0])
        .unwrap();

    assert!((score.log_likelihood - state.subject_log_likelihoods[0]).abs() < 1e-12);
    assert!((score.kappa_log_prior - state.subject_kappa_log_priors[0]).abs() < 1e-12);
    assert!(score.kappa_log_prior.is_finite());
    assert_eq!(
        state
            .kappa_proposal_log_acceptance_ratio(0, 0, 0, &[0.0])
            .unwrap(),
        0.0
    );
}

#[test]
fn iov_controller_exposes_kappa_covariance_and_runs_conditional_mcmc() {
    let mut controller = iov_problem()
        .fit_controller(
            SaemConfig::new()
                .n_chains(2)
                .k1_iterations(2)
                .k2_iterations(0)
                .burn_in(2),
        )
        .unwrap();

    assert_eq!(
        controller.iov_effect_names(),
        Some(["ke".to_string()].as_slice())
    );
    assert_eq!(controller.omega_iov(), Some(&ndarray::array![[0.1]]));
    assert!(controller.kappa_log_prior().is_finite());
    assert_eq!(
        controller.log_posterior(),
        controller.likelihood() + controller.eta_log_prior() + controller.kappa_log_prior()
    );

    controller.step().unwrap();
    assert!(controller.likelihood().is_finite());
    assert!(controller.kappa_log_prior().is_finite());
    assert!(controller.acceptance_rate().is_some());
    assert!(controller
        .kappa_acceptance_rate()
        .is_some_and(|rate| (0.0..=1.0).contains(&rate)));
}

#[test]
fn correlated_random_walk_reuses_one_standard_normal_vector() {
    let proposed =
        correlated_random_walk(&[1.0, 2.0], &[vec![2.0], vec![1.0, 3.0]], &[0.5, -1.0], 0.2)
            .unwrap();

    assert!((proposed[0] - 1.2).abs() < 1e-12);
    assert!((proposed[1] - 1.5).abs() < 1e-12);
    assert!(correlated_random_walk(&[0.0], &[vec![1.0]], &[0.0, 1.0], 1.0).is_err());
}

#[test]
fn eta_block_proposal_uses_covariance_scale_and_adaptation() {
    let lower = vec![vec![1.0], vec![0.8, 0.6]];
    let normals = [[0.5, -1.0], [-0.25, 0.75], [1.2, 0.1], [-0.8, -0.4]];
    let uniforms = [0.2_f64, 0.9, 0.4, 0.7];
    let expected_trace = [
        [0.65, -0.3],
        [0.525, -0.175],
        [0.525, -0.175],
        [0.525, -0.175],
    ];
    let expected_ratios = [
        -0.9451955782312924,
        0.6944515306122447,
        -2.211747363945578,
        -0.4124850340136057,
    ];
    let expected_accepts = [true, true, false, false];
    let expected_scales = [0.55, 0.495];
    let expected_checkpoint_counts = [(2, 2), (0, 2)];
    let log_likelihood =
        |eta: &[f64]| -0.5 * ((eta[0] - 0.3) / 0.5).powi(2) - 0.5 * ((eta[1] + 0.1) / 0.7).powi(2);
    let log_prior = |eta: &[f64]| {
        -0.5 / (1.0 - 0.8_f64.powi(2)) * (eta[0].powi(2) - 1.6 * eta[0] * eta[1] + eta[1].powi(2))
    };

    let mut eta = vec![0.4, -0.2];
    let mut scale = 0.5;
    let mut accepted = 0;
    let mut proposed = 0;
    let mut scale_index = 0;
    for (step, (z, uniform)) in normals.iter().zip(uniforms).enumerate() {
        let proposal = correlated_random_walk(&eta, &lower, z, scale).unwrap();
        let reference = [
            eta[0] + scale * lower[0][0] * z[0],
            eta[1] + scale * (lower[1][0] * z[0] + lower[1][1] * z[1]),
        ];
        assert!((proposal[0] - reference[0]).abs() < 1e-15);
        assert!((proposal[1] - reference[1]).abs() < 1e-15);

        let current_score = SubjectPosteriorScore {
            log_likelihood: log_likelihood(&eta),
            eta_log_prior: log_prior(&eta),
            kappa_log_prior: 0.0,
        };
        let proposed_score = SubjectPosteriorScore {
            log_likelihood: log_likelihood(&proposal),
            eta_log_prior: log_prior(&proposal),
            kappa_log_prior: 0.0,
        };
        let ratio = current_score.log_acceptance_ratio(proposed_score);
        let reference_ratio = proposed_score.log_posterior() - current_score.log_posterior();
        assert!((ratio - reference_ratio).abs() < 1e-15);
        assert!((ratio - expected_ratios[step]).abs() < 1e-12);

        let accept = ratio >= 0.0 || uniform.ln() < ratio;
        assert_eq!(accept, expected_accepts[step]);
        proposed += 1;
        if accept {
            eta = proposal;
            accepted += 1;
        }
        assert!((eta[0] - expected_trace[step][0]).abs() < 1e-12);
        assert!((eta[1] - expected_trace[step][1]).abs() < 1e-12);

        if (step + 1) % 2 == 0 {
            assert_eq!(
                (accepted, proposed),
                expected_checkpoint_counts[scale_index]
            );
            scale = adapt_block_step_size(
                scale,
                accepted as f64 / proposed as f64,
                ETA_BLOCK_TARGET_ACCEPTANCE,
            );
            assert!((scale - expected_scales[scale_index]).abs() < 1e-12);
            scale_index += 1;
            accepted = 0;
            proposed = 0;
        }
    }

    let eta_unchanged = [0.7, -0.3];
    let kappa_0_unchanged = [0.1, 0.2];
    let kappa_1 = correlated_random_walk(
        &[-0.2, 0.4],
        &[vec![0.5], vec![0.1, 0.4]],
        &[-0.5, 0.25],
        0.3,
    )
    .unwrap();
    assert_eq!(eta_unchanged, [0.7, -0.3]);
    assert_eq!(kappa_0_unchanged, [0.1, 0.2]);
    assert!((kappa_1[0] + 0.275).abs() < 1e-12);
    assert!((kappa_1[1] - 0.415).abs() < 1e-12);
}

#[test]
fn eta_block_kernel_runs_before_component_sweep_and_records_diagnostics() {
    let mut state = SaemState::from_problem(
        problem(),
        &SaemConfig::new()
            .n_chains(2)
            .mcmc_iterations(1)
            .eta_block_iterations(2)
            .adapt_interval(50)
            .seed(2024),
    )
    .unwrap();

    state.e_step().unwrap();

    let diagnostics = state.cycle_diagnostics.last().unwrap();
    assert_eq!(diagnostics.eta_block_proposals, 2 * 2 * 2);
    assert_eq!(
        diagnostics.eta_block_accepted + diagnostics.eta_block_rejected,
        diagnostics.eta_block_proposals
    );
    assert_eq!(diagnostics.eta_proposals, 2 * 2 * 2 + 2 * 2 * 2);
    assert_eq!(diagnostics.eta_block_subject_acceptance_rates.len(), 2);
    assert_eq!(
        diagnostics.eta_block_step_sizes_before_adaptation,
        vec![0.5, 0.5]
    );
    assert_eq!(
        diagnostics.eta_block_step_sizes_after_adaptation,
        vec![0.5, 0.5]
    );
}

#[test]
fn controller_exposes_opt_in_eta_block_acceptance_and_scales() {
    let mut controller = problem()
        .fit_controller(
            SaemConfig::new()
                .n_chains(2)
                .eta_block_iterations(1)
                .k1_iterations(1)
                .k2_iterations(0)
                .burn_in(1),
        )
        .unwrap();

    assert_eq!(
        controller.eta_block_step_sizes(),
        Some([0.5, 0.5].as_slice())
    );
    assert_eq!(controller.eta_block_acceptance_rate(), None);
    controller.step().unwrap();
    assert!(controller
        .eta_block_acceptance_rate()
        .is_some_and(|rate| (0.0..=1.0).contains(&rate)));
}

#[test]
fn eta_block_scale_adapts_per_subject_toward_acceptance_target() {
    let mut state = SaemState::from_problem(
        problem(),
        &SaemConfig::new()
            .n_chains(1)
            .eta_block_iterations(1)
            .adapt_interval(1),
    )
    .unwrap();
    assert_eq!(state.eta_block_step_sizes, vec![0.5, 0.5]);

    state.eta_block_adaptation_accept_counts = vec![1, 0];
    state.eta_block_adaptation_proposal_counts = vec![1, 1];
    state.steps_since_adapt = 1;
    state.adapt_proposal_step_sizes();
    assert_eq!(state.eta_block_step_sizes, vec![0.55, 0.45]);
    assert_eq!(state.eta_block_adaptation_accept_counts, vec![0, 0]);
    assert_eq!(state.eta_block_adaptation_proposal_counts, vec![0, 0]);
}

#[test]
fn kappa_block_scale_adapts_per_subject_toward_acceptance_target() {
    let mut state = SaemState::from_problem(
        iov_problem(),
        &SaemConfig::new().n_chains(2).adapt_interval(1),
    )
    .unwrap();
    assert_eq!(state.kappa_proposal_step_sizes, vec![0.5]);

    state.kappa_adaptation_accept_counts[0] = 1;
    state.kappa_adaptation_proposal_counts[0] = 1;
    state.steps_since_adapt = 1;
    state.adapt_proposal_step_sizes();
    assert!((state.kappa_proposal_step_sizes[0] - 0.55).abs() < 1e-12);

    state.kappa_adaptation_accept_counts[0] = 0;
    state.kappa_adaptation_proposal_counts[0] = 1;
    state.steps_since_adapt = 1;
    state.adapt_proposal_step_sizes();
    assert!((state.kappa_proposal_step_sizes[0] - 0.495).abs() < 1e-12);
}

#[test]
fn iov_second_moment_weights_each_occasion_chain_sample_equally() {
    let kappas = vec![
        vec![vec![vec![1.0, 2.0]]],
        vec![vec![vec![3.0, 4.0], vec![5.0, 6.0]]],
    ];

    let covariance = covariance_from_kappas(&kappas).unwrap();

    assert!((covariance[[0, 0]] - 35.0 / 3.0).abs() < 1e-12);
    assert!((covariance[[0, 1]] - 44.0 / 3.0).abs() < 1e-12);
    assert!((covariance[[1, 0]] - 44.0 / 3.0).abs() < 1e-12);
    assert!((covariance[[1, 1]] - 56.0 / 3.0).abs() < 1e-12);
}

#[test]
fn iov_m_step_updates_omega_from_all_occasions() {
    let mut state = SaemState::from_problem(
        iov_problem(),
        &SaemConfig::new()
            .n_chains(2)
            .burn_in(0)
            .omega_sa_max_step(1.0),
    )
    .unwrap();
    state.cycle = 1;
    state.e_step().unwrap();
    for kappas in &mut state.kappas[0] {
        kappas[0][0] = 0.2;
        kappas[1][0] = -0.1;
    }

    state.m_step().unwrap();

    assert!((state.omega_iov.as_ref().unwrap()[[0, 0]] - 0.025).abs() < 1e-12);
    assert!(
        !state
            .cycle_diagnostics
            .last()
            .unwrap()
            .omega_iov_update_rejected
    );
}

#[test]
fn covariance_update_status_drives_iiv_and_iov_cycle_rejection_diagnostics() {
    let config = SaemConfig::new()
        .n_chains(2)
        .burn_in(0)
        .omega_sa_max_step(1.0);
    let mut state = SaemState::from_problem(iov_problem(), &config).unwrap();
    state.cycle = 1;
    state.e_step().unwrap();
    state.iiv_second_moment.fill(f64::NAN);
    state.iov_second_moment.as_mut().unwrap().fill(f64::NAN);

    state.m_step().unwrap();

    let diagnostics = state.cycle_diagnostics.last().unwrap();
    assert!(diagnostics.omega_update_rejected);
    assert!(diagnostics.omega_iov_update_rejected);
}

#[test]
fn iov_second_moment_uses_saem_smoothing_step() {
    let config = SaemConfig::new()
        .n_chains(2)
        .burn_in(0)
        .omega_sa_max_step(1.0)
        .k1_iterations(1)
        .k2_iterations(2);
    let mut state = SaemState::from_problem(iov_problem(), &config).unwrap();
    for kappas in &mut state.kappas[0] {
        kappas[0][0] = 0.2;
        kappas[1][0] = -0.1;
    }
    state.cycle = 1;
    state.m_step().unwrap();

    for kappas in &mut state.kappas[0] {
        kappas[0][0] = 0.2;
        kappas[1][0] = 0.2;
    }
    state.cycle = 3; // first smoothing iteration after K1: γ = 1/2
    state.m_step().unwrap();

    assert!((state.omega_iov.as_ref().unwrap()[[0, 0]] - 0.0325).abs() < 1e-12);
}

#[test]
fn iov_m_step_preserves_fixed_entries_and_positive_definiteness_jointly() {
    let config = SaemConfig::new()
        .n_chains(2)
        .burn_in(0)
        .omega_sa_max_step(1.0);
    let mut state = SaemState::from_problem(configured_iov_problem(), &config).unwrap();
    state.cycle = 1;
    for chain in &mut state.kappas[0] {
        for kappa in chain {
            kappa[0] = 0.3;
            kappa[1] = 1.0;
        }
    }

    state.m_step().unwrap();

    let omega_iov = state.omega_iov.as_ref().unwrap();
    // With fixed b=.20 and c=.05, the exact constrained profile optimum is
    // S11 - 2(c/b)S12 + c²/b + (c²/b²)S22 = .015.
    assert!((omega_iov[[0, 0]] - 0.015).abs() < 1e-12);
    assert_eq!(omega_iov[[0, 1]], 0.05);
    assert_eq!(omega_iov[[1, 0]], 0.05);
    assert_eq!(omega_iov[[1, 1]], 0.20);
    assert!(omega_iov[[0, 0]] * omega_iov[[1, 1]] - omega_iov[[0, 1]].powi(2) > 0.0);
}

#[test]
fn state_uses_declared_initial_omega() {
    let state = SaemState::from_problem(configured_omega_problem(), &SaemConfig::new().n_chains(2))
        .unwrap();

    assert_eq!(state.omega, ndarray::array![[0.25, 0.0], [0.0, 0.5]]);
    assert_eq!(state.proposal_step_sizes, vec![0.25, 0.25 * 2.0_f64.sqrt()]);
}

#[test]
fn individual_parameters_add_eta_in_phi_space() {
    let mut state = SaemState::from_problem(problem(), &SaemConfig::new().n_chains(1)).unwrap();

    let initial = state.individual_parameters(0, 0);
    assert!((initial[0] - 0.2).abs() < 1e-12);
    assert!((initial[1] - 10.0).abs() < 1e-12);

    state.etas[0][0][0] = 2.0_f64.ln();
    state.etas[0][0][1] = 0.5_f64.ln();
    let individual = state.individual_parameters(0, 0);

    assert!((individual[0] - 0.4).abs() < 1e-12);
    assert!((individual[1] - 5.0).abs() < 1e-12);
}

#[test]
fn bounded_transforms_round_trip() {
    let logit = ParameterScale::Logit {
        lower: 0.0,
        upper: 1.0,
    };
    let probit = ParameterScale::Probit {
        lower: 0.0,
        upper: 1.0,
    };

    assert!((phi_to_psi(psi_to_phi(0.25, logit), logit) - 0.25).abs() < 1e-12);
    assert!((phi_to_psi(psi_to_phi(0.25, probit), probit) - 0.25).abs() < 1e-12);
}
