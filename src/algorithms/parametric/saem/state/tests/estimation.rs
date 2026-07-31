use super::*;
#[test]
fn exploration_covariance_cap_prevents_one_draw_rank_one_collapse() {
    fn correlation(omega: &Array2<f64>) -> f64 {
        omega[[0, 1]] / (omega[[0, 0]] * omega[[1, 1]]).sqrt()
    }

    let make_state = |omega_sa_max_step| {
        SaemState::from_problem(
            correlated_omega_problem(),
            &SaemConfig::new()
                .n_chains(1)
                .burn_in(0)
                .omega_sa_max_step(omega_sa_max_step),
        )
        .unwrap()
    };
    let mut guarded = make_state(0.1);
    let mut uncapped = make_state(1.0);
    for state in [&mut guarded, &mut uncapped] {
        state.cycle = 1;
        state.etas[0][0] = vec![2.0, 2.0];
        state.etas[1][0] = vec![-2.0, -2.0];
        state.m_step().unwrap();
        assert!(state.omega[[0, 0]] >= state.initialization.schedule.minimum_variance);
        assert!(state.omega[[1, 1]] >= state.initialization.schedule.minimum_variance);
        assert!(
            state.omega[[0, 0]] * state.omega[[1, 1]] - state.omega[[0, 1]].powi(2) > 0.0,
            "omega: {:?}",
            state.omega
        );
    }

    let guarded_correlation = correlation(&guarded.omega);
    let uncapped_correlation = correlation(&uncapped.omega);
    assert!(guarded_correlation < 0.85);
    assert!(uncapped_correlation > 0.85);
    assert!(uncapped_correlation - guarded_correlation > 0.05);
}

#[test]
fn m_step_preserves_fixed_omega_and_structural_zeros() {
    let mut state = SaemState::from_problem(
        configured_omega_problem(),
        &SaemConfig::new()
            .n_chains(2)
            .burn_in(0)
            .omega_sa_max_step(1.0),
    )
    .unwrap();
    state.cycle = 1;
    for (subject_index, subject_chains) in state.etas.iter_mut().enumerate() {
        let sign = if subject_index == 0 { 1.0 } else { -1.0 };
        for eta in subject_chains {
            eta[0] = sign;
            eta[1] = 2.0 * sign;
        }
    }

    state.m_step().unwrap();

    assert!((state.omega[[0, 0]] - 1.0).abs() < 1e-12);
    assert!((state.omega[[1, 1]] - 0.5).abs() < 1e-12);
    assert_eq!(state.omega[[0, 1]], 0.0);
    assert_eq!(state.omega[[1, 0]], 0.0);
}

#[test]
fn fixed_population_effect_is_not_updated_and_omega_uses_fixed_center() {
    let mut state = SaemState::from_problem(
        fixed_population_iiv_problem(),
        &SaemConfig::new()
            .n_chains(2)
            .burn_in(0)
            .omega_sa_max_step(1.0),
    )
    .unwrap();
    state.cycle = 1;
    for subject_chains in &mut state.etas {
        for eta in subject_chains {
            eta[0] = 2.0_f64.ln();
        }
    }

    state.m_step().unwrap();

    assert!((state.population_parameters[0] - 0.2).abs() < 1e-12);
    assert!(state
        .etas
        .iter()
        .flat_map(|subject_chains| subject_chains.iter())
        .all(|eta| (eta[0] - 2.0_f64.ln()).abs() < 1e-12));
    assert!((state.omega[[0, 0]] - 2.0_f64.ln().powi(2)).abs() < 1e-12);
    let individual = state.individual_parameters(0, 0);
    assert!((individual[0] - 0.4).abs() < 1e-12);
}

#[test]
fn m_step_updates_simple_residual_sigma_from_statrese() {
    let mut state = SaemState::from_problem(
        constant_error_problem(),
        &SaemConfig::new().n_chains(1).burn_in(0),
    )
    .unwrap();
    state.cycle = 1;
    let candidate_sigma = state
        .current_residual_statistics()
        .unwrap()
        .output(0)
        .and_then(|statistic| statistic.sigma())
        .unwrap();
    let expected_sigma = state.initialization.schedule.guarded_residual_sigma(
        state.cycle,
        state.residual_sigmas[0],
        candidate_sigma,
    );

    state.m_step().unwrap();

    assert!((state.residual_sigmas[0] - expected_sigma).abs() < 1e-12);
    assert_eq!(
        state.error_models.get(0),
        Some(&ResidualErrorModel::constant(expected_sigma))
    );
}

#[test]
fn sparse_second_output_reports_only_declared_residual_model() {
    let result = sparse_second_output_problem()
        .fit_with(
            SaemConfig::new()
                .n_chains(1)
                .k1_iterations(1)
                .k2_iterations(0)
                .burn_in(0)
                .compute_map(false),
        )
        .unwrap();

    assert_eq!(result.residual_sigmas().len(), 1);
    assert_eq!(result.residual_error_estimates().len(), 1);
    assert_eq!(result.residual_error_estimates()[0].output, "measured");
    assert_eq!(result.residual_error_estimates()[0].output_index, 1);
    assert_eq!(result.cycle_diagnostics().len(), 1);
    assert_eq!(result.cycle_diagnostics()[0].residual_diagnostics.len(), 1);
    assert_eq!(
        result.cycle_diagnostics()[0].residual_diagnostics[0].output,
        "measured"
    );
    assert_eq!(
        result.cycle_diagnostics()[0].residual_diagnostics[0].output_index,
        1
    );
}

#[test]
fn averaged_sparse_second_output_preserves_index_name_and_arithmetic_mean() {
    let result = sparse_second_output_problem()
        .fit_with(
            SaemConfig::new()
                .n_chains(1)
                .k1_iterations(1)
                .k2_iterations(3)
                .burn_in(0)
                .averaged_iterates(0.75)
                .compute_map(false)
                .seed(71_002),
        )
        .expect("averaged sparse-output fit should complete");

    let metadata = result.estimator_metadata();
    assert!(metadata.average_applied);
    assert_eq!(metadata.averaging_start_cycle, Some(2));
    assert_eq!(metadata.averaged_iterations, 3);
    let estimate = result
        .residual_error_estimates()
        .first()
        .expect("sparse residual estimate");
    assert_eq!(
        (estimate.output_index, estimate.output.as_str()),
        (1, "measured")
    );
    let smoothing = &result.cycle_diagnostics()[1..];
    let expected = smoothing
        .iter()
        .map(|cycle| {
            let residual = cycle
                .residual_error_estimates
                .first()
                .expect("sparse cycle residual");
            assert_eq!(
                (residual.output_index, residual.output.as_str()),
                (1, "measured")
            );
            primary_sigma_parameter(&residual.model)
        })
        .sum::<f64>()
        / smoothing.len() as f64;
    assert!((primary_sigma_parameter(&estimate.model) - expected).abs() < 1e-12);
}

#[test]
fn averaged_multi_output_residuals_preserve_fixed_and_fixed_zero_components() {
    let result = mixed_residual_output_problem()
        .fit_with(
            SaemConfig::new()
                .n_chains(1)
                .k1_iterations(1)
                .k2_iterations(3)
                .burn_in(0)
                .averaged_iterates(0.75)
                .compute_map(false)
                .seed(71_003),
        )
        .expect("averaged mixed-output fit should complete");
    let estimates = result.residual_error_estimates();
    assert_eq!(estimates.len(), 2);
    assert_eq!(
        (estimates[0].output_index, estimates[0].output.as_str()),
        (0, "fixed")
    );
    assert_eq!(estimates[0].model, ResidualErrorModel::constant(0.5));
    assert!(!estimates[0].estimated);
    assert_eq!(
        (estimates[1].output_index, estimates[1].output.as_str()),
        (1, "mixed")
    );
    assert_eq!(estimates[1].combined_additive_estimated, Some(false));
    assert_eq!(estimates[1].combined_proportional_estimated, Some(true));
    let ResidualErrorModel::Combined { a, b } = estimates[1].model else {
        panic!("expected combined residual model");
    };
    assert_eq!(a, 0.0);
    let smoothing = &result.cycle_diagnostics()[1..];
    let expected_b = smoothing
        .iter()
        .map(|cycle| match cycle.residual_error_estimates[1].model {
            ResidualErrorModel::Combined { a, b } => {
                assert_eq!(a, 0.0);
                b
            }
            _ => panic!("expected combined cycle residual model"),
        })
        .sum::<f64>()
        / smoothing.len() as f64;
    assert!((b - expected_b).abs() < 1e-12);
    assert!(result.cycle_diagnostics().iter().all(|cycle| {
        cycle.residual_error_estimates[0].model == ResidualErrorModel::constant(0.5)
    }));
}

#[test]
fn correlated_residual_averaging_preserves_fixed_components_and_rejects_family_changes() {
    let averaged = average_residual_model(
        ResidualErrorModel::correlated_combined(0.3, 0.1, 0.2),
        ResidualErrorModel::correlated_combined(0.5, 0.2, -0.4),
        true,
        [true, true],
        [false, true, true],
        2,
    )
    .unwrap();
    let ResidualErrorModel::CorrelatedCombined { a, b, rho } = averaged else {
        panic!("expected correlated-combined average")
    };
    assert_eq!(a, 0.3);
    assert!((b - 0.15).abs() < 1e-15);
    assert!((rho + 0.1).abs() < 1e-15);
    assert!(average_residual_model(
        averaged,
        ResidualErrorModel::combined(0.3, 0.15),
        true,
        [true, true],
        [true, true, true],
        3,
    )
    .is_err());
}

#[test]
fn population_predictions_match_direct_execution_and_metadata() {
    let result = problem()
        .fit_with(
            SaemConfig::new()
                .n_chains(1)
                .k1_iterations(1)
                .k2_iterations(0)
                .burn_in(1),
        )
        .unwrap();
    let predictions = result.population_predictions(0.25, 0.0).unwrap();
    let expanded = result.data().clone().expand(0.25, 0.0, &[]);

    assert_eq!(predictions.len(), expanded.subjects().len());
    assert_eq!(expanded.subjects()[0].id(), "s1");
    assert_eq!(expanded.subjects()[1].id(), "s2");
    for (subject, actual) in expanded.subjects().iter().zip(&predictions) {
        let expected = result
            .equation()
            .estimate_predictions_dense(subject, result.population_parameters())
            .unwrap();
        assert_prediction_points_equal(actual, &expected);
    }
}

#[test]
fn sparse_output_prediction_expansion_preserves_observed_outputs_only() {
    let result = sparse_second_output_problem()
        .fit_with(
            SaemConfig::new()
                .n_chains(1)
                .k1_iterations(1)
                .k2_iterations(0)
                .burn_in(1),
        )
        .unwrap();

    let population = result.population_predictions(0.25, 0.0).unwrap();
    let conditional = result.conditional_predictions(0.25, 0.0).unwrap();
    for predictions in population.iter().chain(&conditional) {
        assert!(predictions
            .predictions()
            .iter()
            .all(|prediction| prediction.output().as_str() == "measured"));
    }
    let tables = result.tables(0.25, 0.0).unwrap();
    assert!(tables
        .predictions
        .iter()
        .all(|prediction| prediction.output_index == 1));
}

#[test]
fn fixed_zero_latent_conditional_predictions_equal_population_predictions() {
    let result = fixed_no_iiv_problem()
        .fit_with(
            SaemConfig::new()
                .n_chains(1)
                .k1_iterations(1)
                .k2_iterations(0)
                .burn_in(1),
        )
        .unwrap();

    assert!(result.conditional_modes().is_empty());
    let population = result.population_predictions(0.25, 0.0).unwrap();
    let conditional = result.conditional_predictions(0.25, 0.0).unwrap();
    assert_eq!(conditional.len(), population.len());
    for (conditional, population) in conditional.iter().zip(&population) {
        assert_prediction_points_equal(conditional, population);
    }
}

#[test]
fn iov_conditional_predictions_use_each_occasion_kappa_in_order() {
    let mut result = iov_problem()
        .fit_with(
            SaemConfig::new()
                .n_chains(1)
                .k1_iterations(1)
                .k2_iterations(0)
                .burn_in(1),
        )
        .unwrap();
    result.conditional_modes[0].eta.fill(0.0);
    result.conditional_modes[0].kappas[0].values[0] = -0.2;
    result.conditional_modes[0].kappas[1].values[0] = 0.3;

    let actual = result.conditional_predictions(0.25, 0.0).unwrap();
    assert_eq!(actual.len(), 1);
    let expanded = result.data().clone().expand(0.25, 0.0, &[]);
    let subject = &expanded.subjects()[0];
    let mode = &result.conditional_modes()[0];
    let mut expected = pharmsol::simulator::prediction::SubjectPredictions::default();
    expected.set_id(subject.id().clone());
    for (occasion, kappa) in subject.occasions().iter().zip(&mode.kappas) {
        let parameters = occasion_psi(
            result.population_parameters(),
            &result.parameter_scales,
            &result.random_effect_indices,
            &mode.eta,
            &result.iov_effect_indices,
            &kappa.values,
        )
        .unwrap();
        let occasion_subject =
            Subject::from_occasions(subject.id().clone(), vec![occasion.clone()]);
        for prediction in result
            .equation()
            .estimate_predictions_dense(&occasion_subject, &parameters)
            .unwrap()
            .predictions()
            .iter()
            .cloned()
        {
            expected.add_prediction(prediction, occasion.index());
        }
    }
    assert_prediction_points_equal(&actual[0], &expected);
    assert!(actual[0]
        .occasions()
        .windows(2)
        .any(|pair| pair[0] != pair[1]));
    let occasion_predictions = subject
        .occasions()
        .iter()
        .map(|occasion| {
            actual[0]
                .predictions()
                .iter()
                .zip(actual[0].occasions())
                .find(|(prediction, prediction_occasion)| {
                    **prediction_occasion == occasion.index() && prediction.observation().is_some()
                })
                .unwrap()
                .0
                .prediction()
        })
        .collect::<Vec<_>>();
    assert_ne!(occasion_predictions[0], occasion_predictions[1]);
}
