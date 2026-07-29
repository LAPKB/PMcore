use super::*;
#[test]
fn frozen_markov_diagnostic_is_repeatable_and_canonical_result_is_unchanged() {
    use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};

    let base = SaemConfig::new()
        .k1_iterations(100)
        .k2_iterations(50)
        .burn_in(1)
        .n_chains(2)
        .eta_block_iterations(1)
        .compute_map(true)
        .seed(91)
        .averaged_iterates(0.75);
    let diagnostic_config = MarkovSimulationVarianceConfig::new(
        700,
        2,
        12,
        6,
        LugsailConfig::over_lugsail_bartlett(),
        2,
        64 * 1024,
    );
    let disabled = markov_iov_problem().fit_with(base.clone()).unwrap();
    let enabled = markov_iov_problem()
        .fit_with(base.clone().markov_simulation_variance(diagnostic_config))
        .unwrap();
    let repeated = markov_iov_problem()
        .fit_with(base.clone().markov_simulation_variance(diagnostic_config))
        .unwrap();
    let changed_seed = markov_iov_problem()
        .fit_with(
            base.clone()
                .markov_simulation_variance(MarkovSimulationVarianceConfig::new(
                    701,
                    2,
                    12,
                    6,
                    LugsailConfig::over_lugsail_bartlett(),
                    2,
                    64 * 1024,
                )),
        )
        .unwrap();

    assert_eq!(
        enabled.markov_simulation_variance(),
        repeated.markov_simulation_variance()
    );
    assert_ne!(
        enabled.markov_simulation_variance(),
        changed_seed.markov_simulation_variance()
    );
    assert_ne!(
        enabled.markov_simulation_variance().status,
        MarkovSimulationVarianceStatus::Disabled
    );
    assert!(!enabled.markov_simulation_variance().chains.is_empty());
    // One subject, one eta block, one component eta, and two occasion-kappa
    // blocks are attempted in that exact compound-kernel order per retained
    // transition. Warmup attempts are absent from the exported count.
    assert!(enabled
        .markov_simulation_variance()
        .chains
        .iter()
        .all(|chain| chain.proposals == 12 * (1 + 1 + 2)));
    assert_eq!(
        enabled.population_parameters(),
        disabled.population_parameters()
    );
    assert_eq!(enabled.omega(), disabled.omega());
    assert_eq!(enabled.omega_iov(), disabled.omega_iov());
    assert_eq!(
        enabled.residual_error_estimates(),
        disabled.residual_error_estimates()
    );
    assert_eq!(enabled.eta_chain_means(), disabled.eta_chain_means());
    assert_eq!(enabled.kappa_chain_means(), disabled.kappa_chain_means());
    assert!(!enabled.conditional_modes().is_empty());
    assert_eq!(enabled.conditional_modes(), disabled.conditional_modes());
    assert_eq!(
        enabled.information_diagnostics(),
        disabled.information_diagnostics()
    );
    assert_eq!(enabled.cycle_diagnostics(), disabled.cycle_diagnostics());
    assert_eq!(enabled.warnings(), disabled.warnings());
    assert_eq!(enabled.conditional_n2ll(), disabled.conditional_n2ll());
    assert_eq!(enabled.termination_reason(), disabled.termination_reason());
    assert_eq!(
        enabled.population_parameters(),
        changed_seed.population_parameters()
    );
    assert_eq!(enabled.omega(), changed_seed.omega());
    assert_eq!(
        enabled.residual_error_estimates(),
        changed_seed.residual_error_estimates()
    );
    assert_eq!(enabled.eta_chain_means(), changed_seed.eta_chain_means());
    assert_eq!(
        enabled.cycle_diagnostics(),
        changed_seed.cycle_diagnostics()
    );
    assert_eq!(enabled.warnings(), changed_seed.warnings());
    assert_eq!(enabled.conditional_n2ll(), changed_seed.conditional_n2ll());
    let enabled_predictions = enabled.population_predictions(0.0, 0.0).unwrap();
    let disabled_predictions = disabled.population_predictions(0.0, 0.0).unwrap();
    assert_eq!(enabled_predictions.len(), disabled_predictions.len());
    for (actual, expected) in enabled_predictions.iter().zip(&disabled_predictions) {
        assert_prediction_points_equal(actual, expected);
    }
    let enabled_conditional = enabled.conditional_predictions(0.0, 0.0).unwrap();
    let disabled_conditional = disabled.conditional_predictions(0.0, 0.0).unwrap();
    assert_eq!(enabled_conditional.len(), disabled_conditional.len());
    for (actual, expected) in enabled_conditional.iter().zip(&disabled_conditional) {
        assert_prediction_points_equal(actual, expected);
    }
}

#[test]
fn rank_diagnostics_computed_for_multiple_chains_and_iov() {
    use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
    use crate::results::{DiagnosticTraceCoordinate, RankDiagnosticStatus};

    let base = SaemConfig::new()
        .k1_iterations(30)
        .k2_iterations(20)
        .burn_in(1)
        .n_chains(2)
        .eta_block_iterations(1)
        .compute_map(false)
        .seed(91)
        .averaged_iterates(0.75);
    let diag = MarkovSimulationVarianceConfig::new(
        700,
        0,
        12,
        6,
        LugsailConfig::over_lugsail_bartlett(),
        2,
        1024 * 1024,
    );
    let result = markov_iov_problem()
        .fit_with(base.clone().markov_simulation_variance(diag))
        .unwrap();
    let rank = &result.markov_simulation_variance().rank_diagnostics;
    assert_eq!(rank.diagnostic_chains, 2);
    assert_eq!(rank.draws_per_chain, 12);
    assert_eq!(rank.original_chains, 2);
    assert_eq!(rank.status, RankDiagnosticStatus::Available);
    assert!(!rank.traces.is_empty());
    // First trace is a score coordinate.
    assert!(matches!(
        rank.traces[0].trace,
        DiagnosticTraceCoordinate::Score { .. }
    ));
    let score_count = result.information_diagnostics().coordinates.len();
    let eta_count = result
        .eta_chain_means()
        .iter()
        .map(|estimate| estimate.values.len())
        .sum::<usize>();
    let kappa_count = result
        .kappa_chain_means()
        .iter()
        .map(|estimate| estimate.values.len())
        .sum::<usize>();
    assert_eq!(rank.traces.len(), score_count + eta_count + kappa_count);
    for (trace, coordinate) in rank
        .traces
        .iter()
        .take(score_count)
        .zip(&result.information_diagnostics().coordinates)
    {
        assert!(matches!(
            &trace.trace,
            DiagnosticTraceCoordinate::Score { index, .. } if *index == coordinate.index
        ));
    }
    assert!(rank
        .traces
        .iter()
        .skip(score_count)
        .take(eta_count)
        .all(|trace| matches!(trace.trace, DiagnosticTraceCoordinate::Eta { .. })));
    assert!(rank
        .traces
        .iter()
        .skip(score_count + eta_count)
        .all(|trace| matches!(trace.trace, DiagnosticTraceCoordinate::Kappa { .. })));
    assert!(rank.diagnostic_mean_lrv.is_some());
    assert!(rank.operational_lrv.is_some());

    // Repeatability: same seed produces identical rank diagnostics.
    let repeated = markov_iov_problem()
        .fit_with(base.clone().markov_simulation_variance(diag))
        .unwrap();
    assert_eq!(
        result.markov_simulation_variance().rank_diagnostics,
        repeated.markov_simulation_variance().rank_diagnostics
    );

    // Canonical result is unchanged by rank diagnostic presence.
    let disabled = markov_iov_problem().fit_with(base).unwrap();
    assert_eq!(
        result.population_parameters(),
        disabled.population_parameters()
    );
    assert_eq!(result.omega(), disabled.omega());
    assert_eq!(result.conditional_n2ll(), disabled.conditional_n2ll());
    assert_eq!(result.termination_reason(), disabled.termination_reason());
    assert_eq!(result.termination_reason(), Some(&StopReason::MaxCycles));
}

#[test]
fn score_failure_does_not_discard_valid_eta_rank_diagnostics() {
    use crate::results::{
        DiagnosticTraceCoordinate, InformationCoordinateKind, RankDiagnosticStatus,
    };

    let state = SaemState::from_problem(problem(), &SaemConfig::default()).unwrap();
    let traces = vec![
        vec![vec![f64::NAN; 8], vec![f64::NAN; 8]],
        vec![
            vec![1.0, 4.0, 2.0, 3.0, 2.0, 4.0, 1.0, 3.0],
            vec![2.1, 3.1, 1.1, 4.1, 3.1, 1.1, 4.1, 2.1],
        ],
    ];
    let coordinates = vec![
        DiagnosticTraceCoordinate::Score {
            index: 0,
            name: "score".into(),
            kind: InformationCoordinateKind::Population { parameter_index: 0 },
        },
        DiagnosticTraceCoordinate::Eta {
            subject: "1".into(),
            effect_index: 0,
            effect_name: "CL".into(),
        },
    ];
    let diagnostics = state.rank_diagnostics_from_traces(2, &traces, &coordinates);
    assert_eq!(
        diagnostics[0].rank_rhat_status,
        RankDiagnosticStatus::ScoreUnavailable
    );
    assert!(diagnostics[0].rank_rhat.is_none());
    assert_eq!(
        diagnostics[1].rank_rhat_status,
        RankDiagnosticStatus::Available
    );
    assert!(diagnostics[1].rank_rhat.is_some());
}

#[test]
fn multimodal_latent_trace_is_detected_while_mixed_score_trace_passes() {
    use crate::results::{
        DiagnosticTraceCoordinate, InformationCoordinateKind, RankDiagnosticStatus,
    };

    let state = SaemState::from_problem(problem(), &SaemConfig::default()).unwrap();
    let traces = vec![
        vec![
            vec![1.0, 4.0, 2.0, 3.0, 2.0, 4.0, 1.0, 3.0],
            vec![2.1, 3.1, 1.1, 4.1, 3.1, 1.1, 4.1, 2.1],
        ],
        vec![
            vec![-10.0, -9.0, -11.0, -8.0, -9.5, -8.5, -10.5, -7.5],
            vec![8.0, 11.0, 9.0, 10.0, 8.5, 10.5, 7.5, 9.5],
        ],
    ];
    let coordinates = vec![
        DiagnosticTraceCoordinate::Score {
            index: 0,
            name: "score".into(),
            kind: InformationCoordinateKind::Population { parameter_index: 0 },
        },
        DiagnosticTraceCoordinate::Eta {
            subject: "1".into(),
            effect_index: 0,
            effect_name: "CL".into(),
        },
    ];
    let diagnostics = state.rank_diagnostics_from_traces(2, &traces, &coordinates);
    assert_eq!(
        diagnostics[0].rank_rhat_status,
        RankDiagnosticStatus::Available
    );
    assert!(diagnostics[0].rank_rhat.is_some_and(|rhat| rhat < 1.1));
    assert_eq!(
        diagnostics[1].rank_rhat_status,
        RankDiagnosticStatus::Available
    );
    assert!(diagnostics[1].rank_rhat.is_some_and(|rhat| rhat > 1.1));
}

#[test]
fn rank_coordinate_retains_valid_rhats_when_bulk_ess_is_unavailable() {
    use crate::results::{DiagnosticTraceCoordinate, RankDiagnosticStatus};

    let state = SaemState::from_problem(problem(), &SaemConfig::default()).unwrap();
    let traces = vec![vec![vec![1.0, 2.0, 4.0, 3.0], vec![1.5, 2.5, 4.5, 3.5]]];
    let coordinates = vec![DiagnosticTraceCoordinate::Eta {
        subject: "1".into(),
        effect_index: 0,
        effect_name: "CL".into(),
    }];
    let diagnostics = state.rank_diagnostics_from_traces(2, &traces, &coordinates);
    let diagnostic = &diagnostics[0];
    assert!(diagnostic.rank_rhat.is_some());
    assert_eq!(diagnostic.rank_rhat_status, RankDiagnosticStatus::Available);
    assert!(diagnostic.folded_rhat.is_some());
    assert_eq!(
        diagnostic.folded_rhat_status,
        RankDiagnosticStatus::Available
    );
    assert!(diagnostic.bulk_ess.is_none());
    assert!(diagnostic.tau.is_none());
    assert_eq!(
        diagnostic.bulk_ess_status,
        RankDiagnosticStatus::TooFewDraws
    );
    assert_eq!(diagnostic.status, RankDiagnosticStatus::PartialAvailability);
}

#[test]
fn derived_max_rhat_requires_both_rank_and_folded_components() {
    use crate::results::{DiagnosticTraceCoordinate, RankDiagnosticStatus};

    let state = SaemState::from_problem(problem(), &SaemConfig::default()).unwrap();
    let traces = vec![vec![
        vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        vec![2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0],
    ]];
    let coordinates = vec![DiagnosticTraceCoordinate::Eta {
        subject: "1".into(),
        effect_index: 0,
        effect_name: "CL".into(),
    }];

    let diagnostics = state.rank_diagnostics_from_traces(2, &traces, &coordinates);
    let diagnostic = &diagnostics[0];
    assert!(diagnostic.rank_rhat.is_some());
    assert_eq!(diagnostic.rank_rhat_status, RankDiagnosticStatus::Available);
    assert!(diagnostic.folded_rhat.is_none());
    assert_eq!(
        diagnostic.folded_rhat_status,
        RankDiagnosticStatus::ConstantDraws
    );
    assert!(diagnostic.max_rhat.is_none());
    assert_eq!(
        diagnostic.max_rhat_status,
        RankDiagnosticStatus::ConstantDraws
    );
    assert_eq!(worst_valid_max_rhat(&diagnostics), None);
}

#[test]
fn rank_diagnostics_available_when_markov_config_enabled() {
    use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
    use crate::results::RankDiagnosticStatus;

    let base = SaemConfig::new()
        .k1_iterations(100)
        .k2_iterations(50)
        .burn_in(1)
        .n_chains(2)
        .eta_block_iterations(1)
        .compute_map(false)
        .seed(77)
        .averaged_iterates(0.75);
    let diag = MarkovSimulationVarianceConfig::new(
        42,
        0,
        12,
        6,
        LugsailConfig::over_lugsail_bartlett(),
        2,
        1024 * 1024,
    );
    let result = markov_iov_problem()
        .fit_with(base.clone().markov_simulation_variance(diag))
        .unwrap();
    let rank = &result.markov_simulation_variance().rank_diagnostics;
    // Rank diagnostics object is always present when markov config enabled;
    // status reflects whether data supported valid computation.
    assert_eq!(rank.diagnostic_chains, 2);
    assert_eq!(rank.original_chains, 2);
    assert!(!matches!(rank.status, RankDiagnosticStatus::Disabled));
}

#[test]
fn one_diagnostic_chain_retains_markov_lrv_but_rank_is_unavailable() {
    use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
    use crate::results::RankDiagnosticStatus;

    let config = SaemConfig::new()
        .k1_iterations(30)
        .k2_iterations(20)
        .burn_in(1)
        .n_chains(2)
        .eta_block_iterations(1)
        .compute_map(false)
        .seed(93)
        .averaged_iterates(0.75)
        .markov_simulation_variance(MarkovSimulationVarianceConfig::new(
            702,
            0,
            12,
            6,
            LugsailConfig::over_lugsail_bartlett(),
            1,
            1024 * 1024,
        ));
    let result = markov_iov_problem().fit_with(config).unwrap();
    let markov = result.markov_simulation_variance();
    assert_eq!(
        markov.rank_diagnostics.status,
        RankDiagnosticStatus::TooFewChains
    );
    assert_eq!(markov.chains.len(), 1);
    assert!(!markov.lambda.is_empty());
    assert!(markov.rank_diagnostics.operational_lrv.is_some());
    assert!(markov.rank_diagnostics.traces.iter().all(|trace| {
        trace.status == RankDiagnosticStatus::TooFewChains
            && trace.rank_rhat.is_none()
            && trace.bulk_ess.is_none()
    }));
    assert_eq!(result.termination_reason(), Some(&StopReason::MaxCycles));
    assert!(!result.converged());
}

#[test]
fn rank_diagnostics_trace_byte_cap_exceeded_is_reported() {
    use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
    use crate::results::RankDiagnosticStatus;

    let base = SaemConfig::new()
        .k1_iterations(100)
        .k2_iterations(50)
        .burn_in(1)
        .n_chains(2)
        .eta_block_iterations(1)
        .compute_map(false)
        .seed(91)
        .averaged_iterates(0.75);
    let tiny_cap = MarkovSimulationVarianceConfig::new(
        700,
        0,
        12,
        6,
        LugsailConfig::over_lugsail_bartlett(),
        2,
        1, // 1 byte cap — guaranteed to be exceeded
    );
    let result = markov_iov_problem()
        .fit_with(base.clone().markov_simulation_variance(tiny_cap))
        .unwrap();
    let rank = &result.markov_simulation_variance().rank_diagnostics;
    assert_eq!(rank.status, RankDiagnosticStatus::TraceByteCapExceeded);
    assert!(rank.traces.is_empty());
    assert!(rank.diagnostic_mean_lrv.is_none());
    assert!(rank.operational_lrv.is_none());
    assert_eq!(rank.max_trace_bytes, 1);
    assert!(rank.accounted_peak_trace_bytes_required > rank.max_trace_bytes);
    assert_eq!(rank.accounted_peak_trace_bytes_used, 0);
    let markov = result.markov_simulation_variance();
    assert!(matches!(
        markov.status,
        MarkovSimulationVarianceStatus::InvalidConfiguration(_)
    ));
    assert_eq!(markov.lambda_status, markov.status);
    assert_eq!(markov.xi_status, markov.status);
    assert_eq!(markov.simulation_covariance_status, markov.status);
    assert!(markov.chains.is_empty());
    // Canonical result is unchanged.
    assert_eq!(result.termination_reason(), Some(&StopReason::MaxCycles));

    let generous = MarkovSimulationVarianceConfig::new(
        700,
        0,
        12,
        6,
        LugsailConfig::over_lugsail_bartlett(),
        2,
        1024 * 1024,
    );
    let measured = markov_iov_problem()
        .fit_with(base.clone().markov_simulation_variance(generous))
        .unwrap();
    let measured_rank = &measured.markov_simulation_variance().rank_diagnostics;
    let trace_count = measured_rank.traces.len();
    let score_width = measured.markov_simulation_variance().coordinates.len();
    let vec_header = std::mem::size_of::<Vec<f64>>();
    let persistent_bytes = 2 * 12 * trace_count * std::mem::size_of::<f64>()
        + trace_count * 2 * vec_header
        + trace_count * vec_header;
    let score_transient_bytes = score_width * 12 * std::mem::size_of::<f64>() + 12 * vec_header;
    let rank_transient_bytes = 2 * 12 * 8 * std::mem::size_of::<f64>() + 2 * 16 * vec_header;
    let expected_bytes = persistent_bytes + score_transient_bytes.max(rank_transient_bytes);
    assert_eq!(
        measured_rank.accounted_peak_trace_bytes_required,
        expected_bytes
    );
    assert_eq!(
        measured_rank.accounted_peak_trace_bytes_used,
        expected_bytes
    );

    let exact_cap = MarkovSimulationVarianceConfig::new(
        700,
        0,
        12,
        6,
        LugsailConfig::over_lugsail_bartlett(),
        2,
        expected_bytes,
    );
    let exact = markov_iov_problem()
        .fit_with(base.clone().markov_simulation_variance(exact_cap))
        .unwrap();
    assert_eq!(
        exact
            .markov_simulation_variance()
            .rank_diagnostics
            .accounted_peak_trace_bytes_used,
        expected_bytes
    );
    assert!(!exact
        .markov_simulation_variance()
        .rank_diagnostics
        .traces
        .is_empty());

    let under_cap = MarkovSimulationVarianceConfig::new(
        700,
        0,
        12,
        6,
        LugsailConfig::over_lugsail_bartlett(),
        2,
        expected_bytes - 1,
    );
    let rejected = markov_iov_problem()
        .fit_with(base.clone().markov_simulation_variance(under_cap))
        .unwrap();
    assert_eq!(
        rejected
            .markov_simulation_variance()
            .rank_diagnostics
            .status,
        RankDiagnosticStatus::TraceByteCapExceeded
    );
    assert_eq!(
        rejected
            .markov_simulation_variance()
            .rank_diagnostics
            .accounted_peak_trace_bytes_used,
        0
    );

    let overflow = MarkovSimulationVarianceConfig::new(
        700,
        0,
        12,
        6,
        LugsailConfig::over_lugsail_bartlett(),
        usize::MAX / 2 + 1,
        usize::MAX,
    );
    let overflowed = markov_iov_problem()
        .fit_with(base.markov_simulation_variance(overflow))
        .unwrap();
    let overflowed = overflowed.markov_simulation_variance();
    assert_eq!(
        overflowed.rank_diagnostics.status,
        RankDiagnosticStatus::TraceMemoryAccountingOverflow
    );
    assert_eq!(
        overflowed.status,
        MarkovSimulationVarianceStatus::TraceMemoryAccountingOverflow
    );
    assert_eq!(
        overflowed
            .rank_diagnostics
            .accounted_peak_trace_bytes_required,
        0
    );
    assert_eq!(
        overflowed.rank_diagnostics.accounted_peak_trace_bytes_used,
        0
    );
    assert!(overflowed.chains.is_empty());
}

#[test]
fn operational_and_frozen_iov_transitions_preserve_compound_kernel_order() {
    let seed = 0x5eed;
    let mut operational = SaemState::from_problem(
        markov_iov_problem(),
        &SaemConfig::new()
            .n_chains(2)
            .mcmc_iterations(1)
            .eta_block_iterations(1)
            .adapt_interval(50)
            .seed(seed),
    )
    .unwrap();
    let initial_eta_scales = operational.proposal_step_sizes.clone();
    let initial_eta_block_scales = operational.eta_block_step_sizes.clone();
    let initial_kappa_scales = operational.kappa_proposal_step_sizes.clone();
    let mut frozen = FrozenDiagnosticState {
        etas: operational.etas.clone(),
        kappas: operational.kappas.clone(),
    };
    let mut frozen_rng = StdRng::seed_from_u64(seed);
    let mut frozen_counts = vec![(0, 0, 0); operational.initialization.n_chains];

    // This single compound transition is order-sensitive: eta blocks consume
    // the stream first, followed by component etas and then occasion kappas.
    operational
        .frozen_diagnostic_transition(&mut frozen, &mut frozen_rng, &mut frozen_counts, None)
        .unwrap();
    operational.e_step().unwrap();

    assert_eq!(operational.etas, frozen.etas);
    assert_eq!(operational.kappas, frozen.kappas);
    assert_eq!(operational.proposal_step_sizes, initial_eta_scales);
    assert_eq!(operational.eta_block_step_sizes, initial_eta_block_scales);
    assert_eq!(operational.kappa_proposal_step_sizes, initial_kappa_scales);

    let diagnostics = operational.cycle_diagnostics.last().unwrap();
    let frozen_proposals = frozen_counts.iter().map(|count| count.0).sum::<usize>();
    let frozen_accepts = frozen_counts.iter().map(|count| count.1).sum::<usize>();
    let frozen_changes = frozen_counts.iter().map(|count| count.2).sum::<usize>();
    assert_eq!(diagnostics.eta_block_proposals, 2);
    assert_eq!(diagnostics.eta_proposals, 4);
    assert_eq!(diagnostics.kappa_proposals, 4);
    assert_eq!(frozen_proposals, 8);
    assert_eq!(
        frozen_accepts,
        diagnostics.eta_accepted + diagnostics.kappa_accepted
    );
    assert_eq!(frozen_changes, frozen_accepts);
    assert_eq!(
        diagnostics.eta_rejected + diagnostics.kappa_rejected,
        frozen_proposals - frozen_accepts
    );
    assert_eq!(diagnostics.eta_non_finite, 0);
    assert_eq!(diagnostics.kappa_non_finite, 0);

    let operational_continuation = operational.rng.random::<u64>();
    let frozen_continuation = frozen_rng.random::<u64>();
    assert_eq!(operational_continuation, frozen_continuation);
}

#[test]
fn warmup_movement_cannot_satisfy_retained_movement_accounting() {
    let mut counts = [(12, 7, 4), (8, 1, 1)];
    begin_retained_transition_accounting(&mut counts);
    // Retained proposals that are accepted without an actual state change
    // still leave the chain eligible for the exact stuck guard.
    counts[0].0 += 3;
    counts[0].1 += 3;
    assert_eq!(counts, [(3, 3, 0), (0, 0, 0)]);
    let stuck: Vec<_> = counts
        .iter()
        .enumerate()
        .filter_map(|(chain, count)| (count.2 == 0).then_some(chain))
        .collect();
    assert_eq!(stuck, [0, 1]);
}

#[test]
fn no_latent_state_reports_exact_zero_markov_variance() {
    use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};

    let result = fixed_no_iiv_problem()
        .fit_with(
            SaemConfig::new()
                .k1_iterations(30)
                .k2_iterations(20)
                .burn_in(1)
                .compute_map(false)
                .averaged_iterates(0.75)
                .markov_simulation_variance(MarkovSimulationVarianceConfig::new(
                    4,
                    100,
                    12,
                    6,
                    LugsailConfig::over_lugsail_bartlett(),
                    2,
                    1024,
                )),
        )
        .unwrap();
    let diagnostic = result.markov_simulation_variance();
    assert_eq!(
        diagnostic.status,
        MarkovSimulationVarianceStatus::ExactZeroNoLatentState
    );
    assert!(diagnostic.chains.is_empty());
    assert_eq!(
        diagnostic.rank_diagnostics.status,
        RankDiagnosticStatus::NoLatent
    );
    assert!(diagnostic.rank_diagnostics.traces.is_empty());
    assert!(diagnostic
        .lambda
        .iter()
        .flatten()
        .all(|value| *value == 0.0));
    assert!(diagnostic.xi.iter().flatten().all(|value| *value == 0.0));
    assert!(diagnostic
        .simulation_covariance
        .iter()
        .flatten()
        .all(|value| *value == 0.0));
}
