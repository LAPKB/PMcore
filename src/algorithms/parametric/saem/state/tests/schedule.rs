use super::*;
#[test]
fn initialization_builds_initial_objective() {
    let initialization = SaemInitialization::create(&problem(), &SaemConfig::default()).unwrap();

    assert_eq!(
        initialization.initial_population_parameters,
        vec![0.2, 10.0]
    );
    assert_eq!(initialization.initial_subject_log_likelihoods.len(), 2);
    assert!(initialization.initial_negative_log_likelihood.is_finite());
}

#[test]
fn initialization_rejects_estimated_iiv_variance_below_floor() {
    let mut config = SaemConfig::new();
    config.omega_min_variance = 0.3;

    let error = SaemInitialization::create(&configured_omega_problem(), &config)
        .unwrap_err()
        .to_string();

    assert!(error.contains(
            "initial Omega variance for estimated effect 'ke' (0.25) is below configured omega_min_variance (0.3)"
        ));
}

#[test]
fn initialization_rejects_estimated_iov_variance_below_floor() {
    let config = SaemConfig::new().omega_iov_min_variance(0.11);

    let error = SaemInitialization::create(&configured_iov_problem(), &config)
        .unwrap_err()
        .to_string();

    assert!(error.contains(
            "initial Omega_IOV variance for estimated effect 'ke' (0.1) is below configured omega_iov_min_variance (0.11)"
        ));
}

#[test]
fn initialization_floor_exempts_fixed_covariance_diagonals() {
    let problem = EstimationProblem::parametric(one_compartment(), data())
        .parameter(Parameter::log("ke").with_initial(0.2))
        .parameter(Parameter::log("v").with_initial(10.0))
        .omega(Omega::diagonal([("ke", 0.25)]).fixed_variance("v", 0.01))
        .error_model("0", ResidualErrorModel::constant(1.0))
        .build()
        .unwrap();
    let mut config = SaemConfig::new();
    config.omega_min_variance = 0.1;

    let initialization = SaemInitialization::create(&problem, &config).unwrap();

    assert_eq!(initialization.omega.initial()[[0, 0]], 0.25);
    assert_eq!(initialization.omega.initial()[[1, 1]], 0.01);
    assert!(!initialization.omega.estimated_mask()[[1, 1]]);
}

#[test]
fn schedule_counts_real_internal_phases() {
    let config = SaemConfig::new()
        .burn_in(100)
        .k1_iterations(400)
        .k2_iterations(700);
    let schedule = SaemSchedule::from_config(&config);
    let counts = (1..=schedule.total_iterations).fold([0_usize; 3], |mut counts, cycle| {
        match schedule.phase(cycle) {
            SaemPhase::BurnIn => counts[0] += 1,
            SaemPhase::Exploration => counts[1] += 1,
            SaemPhase::Smoothing => counts[2] += 1,
        }
        counts
    });

    assert_eq!(counts, [100, 300, 700]);
    assert_eq!(schedule.total_iterations, 1100);
}

#[test]
fn covariate_omega_cap_applies_only_during_exploration() {
    assert_eq!(
        covariate_omega_update_maximum_fraction(true, SaemPhase::BurnIn, 0.1),
        1.0
    );
    assert_eq!(
        covariate_omega_update_maximum_fraction(true, SaemPhase::Exploration, 0.1),
        0.1
    );
    assert_eq!(
        covariate_omega_update_maximum_fraction(true, SaemPhase::Smoothing, 0.1),
        1.0
    );
    assert_eq!(
        covariate_omega_update_maximum_fraction(false, SaemPhase::Exploration, 0.1),
        1.0
    );
}

#[derive(Debug)]
struct CommonMomentCycle {
    expected_phi: Vec<Vec<f64>>,
    global_second_moment: Array2<f64>,
    beta: Vec<f64>,
    subject_means: Vec<Vec<f64>>,
    covariance_target: Array2<f64>,
    omega: Array2<f64>,
}

fn common_moment_cycle(
    statistics: &mut CovariateSufficientStatistics,
    observed: &CovariateSufficientStatistics,
    gain: f64,
    designs: &[Array2<f64>],
    current_omega: &Array2<f64>,
    omega_specification: &ResolvedOmega,
) -> Result<CommonMomentCycle> {
    statistics.stochastic_update(observed, gain)?;
    let offsets = vec![vec![0.0]; designs.len()];
    let beta = solve_covariate_gls(CovariateGlsProblem {
        design: designs,
        expected_phi: &statistics.expected_phi,
        offset: &offsets,
        omega: current_omega,
    })?;
    let subject_means = designs
        .iter()
        .map(|design| vec![design[[0, 0]] * beta[0] + design[[0, 1]] * beta[1]])
        .collect::<Vec<_>>();
    let covariance_target = subject_centered_omega(
        &statistics.global_second_moment,
        &statistics.expected_phi,
        &subject_means,
    )?;
    let omega = omega_specification
        .update_with_status(current_omega, &covariance_target, 1e-6)?
        .matrix;
    Ok(CommonMomentCycle {
        expected_phi: statistics.expected_phi.clone(),
        global_second_moment: statistics.global_second_moment.clone(),
        beta,
        subject_means,
        covariance_target,
        omega,
    })
}

fn assert_nested_close(actual: &[Vec<f64>], expected: &[Vec<f64>]) {
    assert_eq!(actual.len(), expected.len());
    for (actual_row, expected_row) in actual.iter().zip(expected) {
        assert_eq!(actual_row.len(), expected_row.len());
        for (actual_value, expected_value) in actual_row.iter().zip(expected_row) {
            assert!((actual_value - expected_value).abs() <= 1e-12);
        }
    }
}

#[test]
fn common_gain_raw_moments_are_coherent_cycle_by_cycle() {
    let designs = [-1.0, 0.0, 1.0]
        .into_iter()
        .map(|covariate| ndarray::array![[1.0, covariate]])
        .collect::<Vec<_>>();
    let parameters = [Parameter::log("x")].into_iter().collect();
    let prior =
        ParametricPrior::new(parameters, Some(Omega::diagonal([("x", 1.0)])), None).unwrap();
    let mut current_omega = prior.omega().clone();
    let mut statistics = CovariateSufficientStatistics {
        expected_phi: vec![vec![0.0]; 3],
        global_second_moment: ndarray::array![[1.0]],
    };
    let exploration_observed = CovariateSufficientStatistics::from_subject_chains(&[
        vec![vec![-1.4], vec![-0.6]],
        vec![vec![-0.4], vec![0.4]],
        vec![vec![0.6], vec![1.4]],
    ])
    .unwrap();
    let first_smoothing_observed = CovariateSufficientStatistics::from_subject_chains(&[
        vec![vec![-1.5], vec![-0.5]],
        vec![vec![0.5], vec![1.5]],
        vec![vec![2.5], vec![3.5]],
    ])
    .unwrap();
    let second_smoothing_observed = CovariateSufficientStatistics::from_subject_chains(&[
        vec![vec![-3.0], vec![-1.0]],
        vec![vec![-1.0], vec![1.0]],
        vec![vec![1.0], vec![3.0]],
    ])
    .unwrap();

    let burn = common_moment_cycle(
        &mut statistics,
        &exploration_observed,
        0.0,
        &designs,
        &current_omega,
        prior.resolved_omega(),
    )
    .unwrap();
    assert_eq!(burn.expected_phi, vec![vec![0.0]; 3]);
    assert_eq!(burn.global_second_moment, ndarray::array![[1.0]]);
    assert_eq!(burn.beta, vec![0.0, 0.0]);
    assert_eq!(burn.subject_means, vec![vec![0.0]; 3]);
    assert_eq!(burn.covariance_target, ndarray::array![[1.0]]);
    assert_eq!(burn.omega, ndarray::array![[1.0]]);

    let exploration = common_moment_cycle(
        &mut statistics,
        &exploration_observed,
        1.0,
        &designs,
        &current_omega,
        prior.resolved_omega(),
    )
    .unwrap();
    assert_nested_close(
        &exploration.expected_phi,
        &[vec![-1.0], vec![0.0], vec![1.0]],
    );
    assert!((exploration.global_second_moment[[0, 0]] - 62.0 / 75.0).abs() <= 1e-12);
    assert!((exploration.beta[0] - 0.0).abs() <= 1e-12);
    assert!((exploration.beta[1] - 1.0).abs() <= 1e-12);
    assert_nested_close(&exploration.subject_means, &exploration.expected_phi);
    assert!((exploration.covariance_target[[0, 0]] - 0.16).abs() <= 1e-12);
    assert!((exploration.omega[[0, 0]] - 0.16).abs() <= 1e-12);
    current_omega = exploration.omega.clone();

    let first_smoothing = common_moment_cycle(
        &mut statistics,
        &first_smoothing_observed,
        1.0,
        &designs,
        &current_omega,
        prior.resolved_omega(),
    )
    .unwrap();
    assert_nested_close(
        &first_smoothing.expected_phi,
        &[vec![-1.0], vec![1.0], vec![3.0]],
    );
    assert!((first_smoothing.global_second_moment[[0, 0]] - 47.0 / 12.0).abs() <= 1e-12);
    assert!((first_smoothing.beta[0] - 1.0).abs() <= 1e-12);
    assert!((first_smoothing.beta[1] - 2.0).abs() <= 1e-12);
    assert_nested_close(
        &first_smoothing.subject_means,
        &first_smoothing.expected_phi,
    );
    assert!((first_smoothing.covariance_target[[0, 0]] - 0.25).abs() <= 1e-12);
    assert!((first_smoothing.omega[[0, 0]] - 0.25).abs() <= 1e-12);
    current_omega = first_smoothing.omega.clone();

    let second_smoothing = common_moment_cycle(
        &mut statistics,
        &second_smoothing_observed,
        0.5,
        &designs,
        &current_omega,
        prior.resolved_omega(),
    )
    .unwrap();
    assert_nested_close(
        &second_smoothing.expected_phi,
        &[vec![-1.5], vec![0.5], vec![2.5]],
    );
    assert!((second_smoothing.global_second_moment[[0, 0]] - 91.0 / 24.0).abs() <= 1e-12);
    assert!((second_smoothing.beta[0] - 0.5).abs() <= 1e-12);
    assert!((second_smoothing.beta[1] - 2.0).abs() <= 1e-12);
    assert_nested_close(
        &second_smoothing.subject_means,
        &second_smoothing.expected_phi,
    );
    assert!((second_smoothing.covariance_target[[0, 0]] - 0.875).abs() <= 1e-12);
    assert!((second_smoothing.omega[[0, 0]] - 0.875).abs() <= 1e-12);

    for cycle in [burn, exploration, first_smoothing, second_smoothing] {
        let mean_square = cycle
            .expected_phi
            .iter()
            .map(|row| row[0] * row[0])
            .sum::<f64>()
            / cycle.expected_phi.len() as f64;
        assert!(cycle.global_second_moment[[0, 0]] + 1e-12 >= mean_square);
        assert!(cycle.covariance_target[[0, 0]] >= -1e-12);
    }
}

#[test]
fn coherent_covariance_target_precedes_structured_gem_constraints() {
    let coherent_target = ndarray::array![[0.002, 0.0], [0.0, 0.04]];
    assert!(cholesky_lower(&coherent_target).is_ok());
    let parameters = [Parameter::log("ke"), Parameter::log("v")]
        .into_iter()
        .collect();
    let prior = ParametricPrior::new(
        parameters,
        Some(
            Omega::new()
                .variance("ke", 0.02)
                .fixed_variance("v", 0.04)
                .fixed_covariance("ke", "v", 0.012),
        ),
        None,
    )
    .unwrap();

    let constrained = prior
        .resolved_omega()
        .update_with_status(prior.omega(), &coherent_target, 0.0)
        .unwrap();

    assert_eq!(coherent_target[[0, 0]], 0.002);
    assert!((constrained.matrix[[0, 0]] - 0.0092).abs() <= 1e-10);
    assert_eq!(constrained.matrix[[0, 1]], 0.012);
    assert_eq!(constrained.matrix[[1, 1]], 0.04);
    assert_ne!(constrained.matrix, coherent_target);
}

#[test]
fn covariate_update_uses_common_moments_and_no_second_smoothing_gain() {
    let mut statistics =
        CovariateSufficientStatistics::from_subject_chains(&[vec![vec![0.0], vec![2.0]]]).unwrap();
    let exploration_observed =
        CovariateSufficientStatistics::from_subject_chains(&[vec![vec![2.0], vec![4.0]]]).unwrap();
    statistics
        .stochastic_update(&exploration_observed, 1.0)
        .unwrap();
    assert_eq!(statistics.expected_phi, vec![vec![3.0]]);
    assert_eq!(statistics.global_second_moment, ndarray::array![[10.0]]);
    let exploration_variance = statistics.global_second_moment[[0, 0]]
        - statistics.expected_phi[0][0] * statistics.expected_phi[0][0];
    let exploration_candidate = ndarray::array![[exploration_variance]];
    assert_eq!(exploration_candidate, ndarray::array![[1.0]]);

    let parameters = [Parameter::log("x")].into_iter().collect();
    let prior =
        ParametricPrior::new(parameters, Some(Omega::diagonal([("x", 0.25)])), None).unwrap();
    let exploration = prior
        .resolved_omega()
        .update_with_status_and_max_fraction(
            prior.omega(),
            &exploration_candidate,
            0.0,
            covariate_omega_update_maximum_fraction(true, SaemPhase::Exploration, 0.1),
        )
        .unwrap();
    assert!((exploration.matrix[[0, 0]] - 0.325).abs() <= 1e-12);

    let smoothing_observed =
        CovariateSufficientStatistics::from_subject_chains(&[vec![vec![4.0], vec![6.0]]]).unwrap();
    statistics
        .stochastic_update(&smoothing_observed, 0.5)
        .unwrap();
    assert_eq!(statistics.expected_phi, vec![vec![4.0]]);
    assert_eq!(statistics.global_second_moment, ndarray::array![[18.0]]);
    let smoothing_variance = statistics.global_second_moment[[0, 0]]
        - statistics.expected_phi[0][0] * statistics.expected_phi[0][0];
    let smoothing_candidate = ndarray::array![[smoothing_variance]];
    assert_eq!(smoothing_candidate, ndarray::array![[2.0]]);

    let smoothing = prior
        .resolved_omega()
        .update_with_status(&exploration.matrix, &smoothing_candidate, 0.0)
        .unwrap();
    assert_eq!(smoothing.matrix, smoothing_candidate);
}

#[test]
fn covariate_state_m_step_caps_exploration_and_does_not_resmooth_omega() {
    let config = SaemConfig::new()
        .n_chains(2)
        .mcmc_iterations(1)
        .burn_in(1)
        .k1_iterations(2)
        .k2_iterations(2)
        .omega_sa_max_step(0.1)
        .compute_map(false);
    let mut state = SaemState::from_problem(fixed_covariate_iiv_problem(), &config).unwrap();

    for subject_chains in &mut state.etas {
        subject_chains[0][0] = 2.0;
        subject_chains[1][0] = -2.0;
    }
    state.cycle = 2;
    assert_eq!(
        state.initialization.schedule.phase(state.cycle),
        SaemPhase::Exploration
    );
    assert_eq!(
        state
            .initialization
            .schedule
            .stochastic_approximation_step(state.cycle),
        1.0
    );
    state.m_step().unwrap();

    assert!((state.iiv_second_moment[[0, 0]] - 4.0).abs() <= 1e-12);
    assert!((state.omega[[0, 0]] - 1.3).abs() <= 1e-12);

    for subject_chains in &mut state.etas {
        subject_chains[0][0] = 4.0;
        subject_chains[1][0] = -4.0;
    }
    state.cycle = 4;
    assert_eq!(
        state.initialization.schedule.phase(state.cycle),
        SaemPhase::Smoothing
    );
    assert_eq!(
        state
            .initialization
            .schedule
            .stochastic_approximation_step(state.cycle),
        0.5
    );
    state.m_step().unwrap();

    // The common raw history moves from variance 4 toward 16 with gain 0.5,
    // giving 10. Omega installs that coherent target directly. Applying the
    // smoothing gain a second time would instead leave Omega below 10.
    assert!((state.iiv_second_moment[[0, 0]] - 10.0).abs() <= 1e-12);
    assert!((state.omega[[0, 0]] - 10.0).abs() <= 1e-12);
}

#[test]
fn schedule_splits_burn_in_exploration_and_smoothing() {
    let config = SaemConfig::new()
        .k1_iterations(300)
        .k2_iterations(100)
        .burn_in(5);
    let schedule = SaemSchedule::from_config(&config);

    assert_eq!(schedule.pure_burn_in, 5);
    assert_eq!(schedule.exploration_iterations, 295);
    assert_eq!(schedule.smoothing_iterations, 100);
    assert_eq!(schedule.total_iterations, 400);
    assert_eq!(schedule.variance_floor_iterations, 150);
    assert_eq!(schedule.minimum_residual_sigma, 1e-6);
    assert_eq!(schedule.stochastic_approximation_step(1), 0.0);
    assert_eq!(schedule.stochastic_approximation_step(6), 1.0);
    assert_eq!(schedule.stochastic_approximation_step(301), 1.0);
    assert_eq!(schedule.stochastic_approximation_step(302), 0.5);
    assert_eq!(schedule.covariance_step(1), 0.1);
    assert_eq!(schedule.covariance_step(6), 0.1);
    assert_eq!(schedule.covariance_step(300), 0.1);
    assert_eq!(schedule.covariance_step(301), 1.0);
    assert_eq!(schedule.covariance_step(302), 0.5);
    assert!(!schedule.covariance_update_active(5));
    assert!(schedule.covariance_update_active(6));
    assert_eq!(schedule.guarded_residual_sigma(1, 1.0, 0.1), 0.97);
    assert_eq!(schedule.guarded_residual_sigma(151, 1.0, 0.1), 0.1);
    assert_eq!(schedule.guarded_residual_sigma(151, 1.0, 0.0), 1e-6);
}

#[test]
fn averaged_schedule_uses_alpha_only_during_smoothing() {
    let schedule = SaemSchedule::from_config(
        &SaemConfig::new()
            .k1_iterations(3)
            .burn_in(1)
            .k2_iterations(4)
            .averaged_iterates(0.75),
    );
    assert_eq!(schedule.stochastic_approximation_step(1), 0.0);
    assert_eq!(schedule.stochastic_approximation_step(2), 1.0);
    assert_eq!(schedule.stochastic_approximation_step(3), 1.0);
    assert_eq!(schedule.stochastic_approximation_step(4), 1.0);
    assert_eq!(
        schedule.stochastic_approximation_step(5),
        2.0_f64.powf(-0.75)
    );
    assert_eq!(
        schedule.stochastic_approximation_step(7),
        4.0_f64.powf(-0.75)
    );
}

#[test]
fn averaged_result_uses_only_completed_smoothing_iterates() {
    let config = SaemConfig::new()
        .k1_iterations(2)
        .burn_in(1)
        .k2_iterations(3)
        .averaged_iterates(0.75)
        .compute_map(false)
        .seed(9981);
    let result = problem().fit_with(config).unwrap();
    let metadata = result.estimator_metadata();
    assert!(metadata.average_applied);
    assert_eq!(metadata.averaging_start_cycle, Some(3));
    assert_eq!(metadata.averaged_iterations, 3);
    assert_eq!(result.termination_reason(), Some(&StopReason::MaxCycles));

    let smoothing = &result.cycle_diagnostics()[2..];
    for parameter_index in 0..result.population_parameters().len() {
        if !result.estimated_parameters()[parameter_index] {
            continue;
        }
        let expected = smoothing
            .iter()
            .map(|cycle| {
                population_phi(&cycle.population_parameters, result.parameter_scales()).unwrap()
                    [parameter_index]
            })
            .sum::<f64>()
            / smoothing.len() as f64;
        let installed = population_phi(result.population_parameters(), result.parameter_scales())
            .unwrap()[parameter_index];
        assert!((installed - expected).abs() < 1e-12);
    }
    for row in 0..result.omega().nrows() {
        for col in 0..result.omega().ncols() {
            let expected = smoothing
                .iter()
                .map(|cycle| cycle.omega[[row, col]])
                .sum::<f64>()
                / smoothing.len() as f64;
            assert!((result.omega()[[row, col]] - expected).abs() < 1e-12);
        }
    }
    cholesky_lower(result.omega()).unwrap();
}

#[test]
fn averaged_iov_installation_is_canonical_and_preserves_latent_coordinates() {
    let config = SaemConfig::new()
        .n_chains(2)
        .mcmc_iterations(2)
        .k1_iterations(1)
        .k2_iterations(3)
        .burn_in(0)
        .averaged_iterates(0.75)
        .compute_map(false)
        .seed(71_004);
    let mut state = SaemState::from_problem(configured_iov_problem(), &config)
        .expect("averaged IOV state should initialize");
    while matches!(state.status, Status::Continue) {
        state.step().expect("averaged IOV cycle should complete");
    }
    let cycle_records = state.cycle_diagnostics.clone();
    let smoothing = &cycle_records[1..];
    let terminal_phi = population_phi(
        &state.population_parameters,
        &state.initialization.parameter_scales,
    )
    .expect("terminal population phi should be valid");
    let terminal_absolute_phi = state
        .etas
        .iter()
        .map(|chains| {
            chains
                .iter()
                .map(|eta| {
                    state
                        .initialization
                        .random_effect_indices
                        .iter()
                        .enumerate()
                        .map(|(eta_index, parameter_index)| {
                            terminal_phi[*parameter_index] + eta[eta_index]
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let terminal_kappas = state.kappas.clone();
    let average = state
        .iterate_average
        .clone()
        .expect("completed smoothing average");

    let metadata = state
        .install_iterate_average()
        .expect("averaged IOV state should install");
    assert!(metadata.average_applied);
    assert_eq!(metadata.averaging_start_cycle, Some(2));
    assert_eq!(metadata.averaged_iterations, 3);
    assert_eq!(state.cycle_diagnostics, cycle_records);
    assert_eq!(state.kappas, terminal_kappas);

    let installed_phi = population_phi(
        &state.population_parameters,
        &state.initialization.parameter_scales,
    )
    .expect("installed population phi should be valid");
    assert_eq!(installed_phi, average.population_phi);
    for (subject_index, chains) in state.etas.iter().enumerate() {
        for (chain_index, eta) in chains.iter().enumerate() {
            for (eta_index, parameter_index) in state
                .initialization
                .random_effect_indices
                .iter()
                .copied()
                .enumerate()
            {
                assert!(
                    (installed_phi[parameter_index] + eta[eta_index]
                        - terminal_absolute_phi[subject_index][chain_index][eta_index])
                        .abs()
                        < 1e-14
                );
            }
        }
    }

    let omega_iov = state.omega_iov.as_ref().expect("installed Omega_IOV");
    let iov_specification = state
        .initialization
        .omega_iov
        .as_ref()
        .expect("IOV specification");
    assert_eq!(omega_iov, &average.omega_iov.expect("averaged Omega_IOV"));
    for row in 0..omega_iov.nrows() {
        for col in 0..omega_iov.ncols() {
            let expected = if iov_specification.estimated_mask()[[row, col]] {
                smoothing
                    .iter()
                    .map(|cycle| {
                        cycle
                            .omega_iov
                            .as_ref()
                            .expect("smoothing cycle should retain Omega_IOV")[[row, col]]
                    })
                    .sum::<f64>()
                    / smoothing.len() as f64
            } else {
                iov_specification.initial()[[row, col]]
            };
            assert!((omega_iov[[row, col]] - expected).abs() < 1e-12);
        }
    }

    let n_chains = state.initialization.n_chains as f64;
    let mut direct_likelihoods = vec![0.0; state.initialization.subject_ids.len()];
    let mut direct_eta_priors = vec![0.0; state.initialization.subject_ids.len()];
    let mut direct_kappa_priors = vec![0.0; state.initialization.subject_ids.len()];
    for subject_index in 0..state.initialization.subject_ids.len() {
        for chain_index in 0..state.initialization.n_chains {
            let score = state
                .score_subject_latents(
                    subject_index,
                    &state.etas[subject_index][chain_index],
                    &state.kappas[subject_index][chain_index],
                )
                .expect("installed latent score should be directly calculable");
            direct_likelihoods[subject_index] += score.log_likelihood / n_chains;
            direct_eta_priors[subject_index] += score.eta_log_prior / n_chains;
            direct_kappa_priors[subject_index] += score.kappa_log_prior / n_chains;
        }
    }
    assert_eq!(state.subject_log_likelihoods, direct_likelihoods);
    assert_eq!(state.subject_log_priors, direct_eta_priors);
    assert_eq!(state.subject_kappa_log_priors, direct_kappa_priors);
    assert_eq!(
        state.negative_log_likelihood,
        negative_log_likelihood(&direct_likelihoods)
    );
}
