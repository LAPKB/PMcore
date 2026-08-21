use super::*;

#[derive(Debug, Clone)]
struct SaemIterateAverage {
    population_phi: Vec<f64>,
    covariate_betas: Option<Vec<f64>>,
    omega: Array2<f64>,
    omega_iov: Option<Array2<f64>>,
    residual_model_width: usize,
    residual_models: Vec<(usize, ResidualErrorModel)>,
    start_cycle: usize,
    count: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct KernelCounters {
    proposals: usize,
    accepted: usize,
    rejected: usize,
    non_finite: usize,
}

struct SaemixMapDistribution {
    mode: Vec<f64>,
    covariance: Array2<f64>,
    lower: Vec<Vec<f64>>,
}

// ─── Operational convergence lifecycle ────────────────────────────────────
//
// Result types live in `crate::results::fit_result`.
// `OperationalConvergenceConfig` is the source of truth for settings.

/// Domain-separation constant for deterministic per-checkpoint seeds.
///
/// Its fixed bytes are combined with the SAEM seed via wrapping addition.
const OPERATIONAL_CHECKPOINT_SEED_DOMAIN: u64 = 0x4E31_4F50_4352_4954;

/// Per-cycle SAEM estimation state.
///
/// MCMC chains, stochastic-approximation sufficient statistics, and the
/// current population / omega / sigma estimates are updated in-place.
#[derive(Debug)]
pub(crate) struct SaemState<E: Equation> {
    equation: E,
    data: Data,
    error_models: ParametricErrorModels,
    config: SaemConfig,
    pub(crate) initialization: SaemInitialization,
    cycle: usize,
    status: Status,
    numerical_failure: Option<NumericalFailure>,
    etas: Vec<Vec<Vec<f64>>>,
    kappas: Vec<Vec<Vec<Vec<f64>>>>,
    population_parameters: Vec<f64>,
    omega: Array2<f64>,
    omega_iov: Option<Array2<f64>>,
    iiv_second_moment: Array2<f64>,
    iov_second_moment: Option<Array2<f64>>,
    sufficient_statistics: PhiSufficientStatistics,
    covariate_statistics: Option<CovariateSufficientStatistics>,
    subject_mu_phi: Option<Vec<Vec<f64>>>,
    covariate_model: Option<CovariateModel>,
    residual_statistics: ResidualSufficientStatistics,
    residual_sigmas: Vec<f64>,
    information: InformationRecursion,
    proposal_step_sizes: Vec<f64>,
    eta_block_step_sizes: Vec<f64>,
    saemix_subset_step_sizes: Vec<Vec<f64>>,
    kappa_proposal_step_sizes: Vec<f64>,
    mcmc_iterations: usize,
    eta_block_iterations: usize,
    saemix_mcmc: Option<SaemixMcmcConfig>,
    adapt_interval: usize,
    residual_optimizer_max_iterations: usize,
    compute_map: bool,
    map_max_iterations: usize,
    map_sd_tolerance: f64,
    map_initial_step: f64,
    steps_since_adapt: usize,
    adaptation_accept_counts: Vec<usize>,
    adaptation_proposal_counts: Vec<usize>,
    eta_block_adaptation_accept_counts: Vec<usize>,
    eta_block_adaptation_proposal_counts: Vec<usize>,
    kappa_adaptation_accept_counts: Vec<usize>,
    kappa_adaptation_proposal_counts: Vec<usize>,
    rng: StdRng,
    subject_log_likelihoods: Vec<f64>,
    subject_log_priors: Vec<f64>,
    subject_kappa_log_priors: Vec<f64>,
    last_log_acceptance_ratios: Vec<f64>,
    last_acceptance_rate: Option<f64>,
    last_eta_block_acceptance_rate: Option<f64>,
    last_kappa_acceptance_rate: Option<f64>,
    last_rejected_proposals: Option<usize>,
    last_non_finite_proposals: Option<usize>,
    last_parameter_acceptance_rates: Vec<f64>,
    cycle_diagnostics: Vec<SaemCycleDiagnostics>,
    negative_log_likelihood: f64,
    iterate_average: Option<SaemIterateAverage>,
    operational_settings: Option<OperationalConvergenceConfig>,
    operational_diagnostics: OperationalConvergenceDiagnostics,
}

impl<E: Equation> SaemState<E> {
    pub(crate) fn from_problem(
        problem: EstimationProblem<E, Parametric>,
        config: &SaemConfig,
    ) -> Result<Self> {
        let mut initialization = SaemInitialization::create(&problem, config)?;
        let EstimationProblem {
            model,
            data,
            error_models,
            ..
        } = problem;
        // Capture immutable initial residual values and estimated masks before
        // any SAEM cycle modifies them.
        let mut initial_residual_values = Vec::new();
        let mut initial_residual_estimated = Vec::new();
        for (outeq, model) in error_models.models().iter() {
            let estimate = error_models.is_estimated(outeq);
            let combined = error_models.combined_component_estimated(outeq);
            let correlated = error_models.correlated_combined_component_estimated(outeq);
            let (additive, proportional, correlation) =
                if matches!(model, ResidualErrorModel::CorrelatedCombined { .. }) {
                    (correlated[0], correlated[1], Some(correlated[2]))
                } else {
                    (combined[0], combined[1], None)
                };
            let components = crate::results::parametric_output::residual_components(
                *model,
                estimate,
                Some(additive),
                Some(proportional),
                correlation,
            );
            initial_residual_values.push(components.iter().map(|c| c.1).collect());
            initial_residual_estimated.push(components.iter().map(|c| c.2).collect());
        }
        initialization.initial_residual_values = initial_residual_values;
        initialization.initial_residual_estimated = initial_residual_estimated;
        Ok(Self::new(
            model.equation,
            data,
            error_models,
            initialization,
            config,
        ))
    }

    pub(crate) fn new(
        equation: E,
        data: Data,
        error_models: ParametricErrorModels,
        initialization: SaemInitialization,
        config: &SaemConfig,
    ) -> Self {
        let n_random_effects = initialization.random_effect_indices.len();
        let etas = zero_etas(
            initialization.subject_ids.len(),
            initialization.n_chains,
            n_random_effects,
        );
        let kappas = zero_kappas(
            &initialization.occasion_counts,
            initialization.n_chains,
            initialization.iov_effect_indices.len(),
        );
        let population_parameters = initialization.initial_population_parameters.clone();
        let omega = initialization.omega.initial().clone();
        let iiv_second_moment = omega.clone();
        let omega_iov = initialization
            .omega_iov
            .as_ref()
            .map(|omega| omega.initial().clone());
        let iov_second_moment = omega_iov.clone();
        let initial_subject_phi = zero_eta_subject_phi(&population_parameters, &initialization)
            .expect("initial population parameters should produce valid phi statistics");
        let mut sufficient_statistics =
            PhiSufficientStatistics::from_subject_phi(&initial_subject_phi)
                .expect("initial phi statistics should be valid");
        for (eta_row, parameter_row) in initialization.random_effect_indices.iter().enumerate() {
            for (eta_col, parameter_col) in initialization.random_effect_indices.iter().enumerate()
            {
                sufficient_statistics.second_moment[[*parameter_row, *parameter_col]] +=
                    omega[[eta_row, eta_col]];
            }
        }
        let subject_mu_phi = initialization.initial_subject_mu_phi.clone();
        let covariate_model = initialization.covariate_model.clone();
        let covariate_statistics = subject_mu_phi.as_ref().map(|means| {
            let expected_phi = means
                .iter()
                .map(|mean| {
                    initialization
                        .random_effect_indices
                        .iter()
                        .map(|index| mean[*index])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut global_second_moment = Array2::zeros((n_random_effects, n_random_effects));
            for mean in &expected_phi {
                for row in 0..n_random_effects {
                    for column in 0..n_random_effects {
                        global_second_moment[[row, column]] +=
                            mean[row] * mean[column] / expected_phi.len() as f64;
                    }
                }
            }
            global_second_moment += &omega;
            CovariateSufficientStatistics {
                expected_phi,
                global_second_moment,
            }
        });
        let subject_log_priors = eta_log_priors(&etas, &omega, 0)
            .expect("validated initial omega should produce finite eta priors");
        let subject_kappa_log_priors = omega_iov
            .as_ref()
            .map(|omega| {
                kappas
                    .iter()
                    .map(|subject_chains| {
                        subject_chains[0]
                            .iter()
                            .map(|kappa| eta_log_prior_from_omega(kappa, omega))
                            .collect::<Result<Vec<_>>>()
                            .map(|priors| priors.into_iter().sum())
                    })
                    .collect::<Result<Vec<_>>>()
                    .expect("validated initial omega_iov should produce finite kappa priors")
            })
            .unwrap_or_else(|| vec![0.0; initialization.subject_ids.len()]);
        let residual_statistics = ResidualSufficientStatistics::zero(error_models.models().len());
        let residual_sigmas = primary_sigma_parameters(error_models.models());
        let proposal_step_sizes = initial_proposal_step_sizes(&omega, config.rw_init);
        let eta_block_step_sizes = if config.eta_block_iterations > 0 {
            vec![config.rw_init; initialization.subject_ids.len()]
        } else {
            Vec::new()
        };
        let saemix_subset_step_sizes = if config
            .saemix_mcmc
            .is_some_and(|policy| policy.iterations[2] > 0)
        {
            proposal_step_sizes
                .iter()
                .map(|step| vec![*step; n_random_effects])
                .collect()
        } else {
            Vec::new()
        };
        let kappa_proposal_step_sizes = omega_iov
            .as_ref()
            .map(|_| vec![config.rw_init; initialization.subject_ids.len()])
            .unwrap_or_default();
        let mcmc_iterations = config.mcmc_iterations;
        let eta_block_iterations = config.eta_block_iterations;
        let adapt_interval = config.adapt_interval;
        let steps_since_adapt = 0;
        let adaptation_accept_counts = vec![0; n_random_effects];
        let adaptation_proposal_counts = vec![0; n_random_effects];
        let eta_block_adaptation_accept_counts = vec![0; eta_block_step_sizes.len()];
        let eta_block_adaptation_proposal_counts = vec![0; eta_block_step_sizes.len()];
        let kappa_adaptation_accept_counts = vec![0; initialization.subject_ids.len()];
        let kappa_adaptation_proposal_counts = vec![0; initialization.subject_ids.len()];
        let rng = StdRng::seed_from_u64(config.seed);
        let last_log_acceptance_ratios = vec![0.0; initialization.subject_ids.len()];
        let last_acceptance_rate = None;
        let last_parameter_acceptance_rates = vec![0.0; n_random_effects];
        let covariate_effect_names = covariate_model
            .as_ref()
            .map(|model| {
                model
                    .estimates()
                    .iter()
                    .map(|estimate| estimate.name().to_string())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let covariate_estimated = covariate_model
            .as_ref()
            .map(|model| {
                model
                    .estimates()
                    .iter()
                    .map(|estimate| estimate.estimated())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let information_layout = InformationLayout::new(
            &initialization.parameter_names,
            &initialization.estimated_parameters,
            &covariate_effect_names,
            &covariate_estimated,
            &initialization.random_effect_names,
            initialization.omega.structural_mask(),
            initialization.omega.estimated_mask(),
            &initialization.iov_effect_names,
            initialization
                .omega_iov
                .as_ref()
                .map(|omega| omega.structural_mask()),
            initialization
                .omega_iov
                .as_ref()
                .map(|omega| omega.estimated_mask()),
            &error_models,
        )
        .expect("validated SAEM metadata must produce an information layout");
        let mut information = InformationRecursion::new(information_layout);
        let has_non_iiv_population =
            initialization
                .estimated_parameters
                .iter()
                .enumerate()
                .any(|(index, estimated)| {
                    *estimated && !initialization.random_effect_indices.contains(&index)
                });
        let has_non_iiv_covariate = covariate_model.as_ref().is_some_and(|model| {
            model
                .estimates()
                .iter()
                .enumerate()
                .any(|(index, estimate)| {
                    estimate.estimated()
                        && !initialization
                            .random_effect_indices
                            .contains(&model.parameter_indices()[index])
                })
        });
        if has_non_iiv_population || has_non_iiv_covariate {
            information.mark_unavailable(InformationStatus::Unsupported(
                "structural observation sensitivities are unavailable for estimated non-IIV population or covariate coordinates"
                    .to_string(),
            ));
        }

        Self {
            equation,
            data,
            error_models,
            config: config.clone(),
            etas,
            kappas,
            population_parameters,
            omega,
            omega_iov,
            iiv_second_moment,
            iov_second_moment,
            sufficient_statistics,
            covariate_statistics,
            subject_mu_phi,
            covariate_model,
            residual_statistics,
            residual_sigmas,
            information,
            proposal_step_sizes,
            eta_block_step_sizes,
            saemix_subset_step_sizes,
            kappa_proposal_step_sizes,
            mcmc_iterations,
            eta_block_iterations,
            saemix_mcmc: config.saemix_mcmc,
            adapt_interval,
            residual_optimizer_max_iterations: config.residual_optimizer_max_iterations,
            compute_map: config.compute_map,
            map_max_iterations: config.map_max_iterations,
            map_sd_tolerance: config.map_sd_tolerance,
            map_initial_step: config.map_initial_step,
            steps_since_adapt,
            adaptation_accept_counts,
            adaptation_proposal_counts,
            eta_block_adaptation_accept_counts,
            eta_block_adaptation_proposal_counts,
            kappa_adaptation_accept_counts,
            kappa_adaptation_proposal_counts,
            rng,
            subject_log_likelihoods: initialization.initial_subject_log_likelihoods.clone(),
            subject_log_priors,
            subject_kappa_log_priors,
            last_log_acceptance_ratios,
            last_acceptance_rate,
            last_eta_block_acceptance_rate: None,
            last_kappa_acceptance_rate: None,
            last_rejected_proposals: None,
            last_non_finite_proposals: None,
            last_parameter_acceptance_rates,
            cycle_diagnostics: Vec::with_capacity(initialization.schedule.total_iterations),
            negative_log_likelihood: initialization.initial_negative_log_likelihood,
            iterate_average: None,
            operational_settings: config.operational_convergence,
            operational_diagnostics: OperationalConvergenceDiagnostics {
                config: config.operational_convergence,
                ..OperationalConvergenceDiagnostics::default()
            },
            initialization,
            cycle: 0,
            status: Status::Continue,
            numerical_failure: None,
        }
    }

    fn e_step(&mut self) -> Result<()> {
        let mut eta_accepted = 0usize;
        let mut eta_rejected = 0usize;
        let mut eta_non_finite = 0usize;
        let mut eta_proposed = 0usize;
        let mut eta_block_accepted = 0usize;
        let mut eta_block_rejected = 0usize;
        let mut eta_block_non_finite = 0usize;
        let mut eta_block_proposed = 0usize;
        let mut kappa_accepted = 0usize;
        let mut kappa_rejected = 0usize;
        let mut kappa_non_finite = 0usize;
        let mut kappa_proposed = 0usize;
        let eta_step_sizes_before = self.proposal_step_sizes.clone();
        let eta_block_step_sizes_before = self.eta_block_step_sizes.clone();
        let kappa_step_sizes_before = self.kappa_proposal_step_sizes.clone();
        let kappa_subject_count = if self.omega_iov.is_some() {
            self.initialization.subject_ids.len()
        } else {
            0
        };
        let mut kappa_subject_accept_counts = vec![0usize; kappa_subject_count];
        let mut kappa_subject_proposal_counts = vec![0usize; kappa_subject_count];
        let eta_block_subject_count = if self.eta_block_iterations > 0 {
            self.initialization.subject_ids.len()
        } else {
            0
        };
        let mut eta_block_subject_accept_counts = vec![0usize; eta_block_subject_count];
        let mut eta_block_subject_proposal_counts = vec![0usize; eta_block_subject_count];
        let n_parameters = self.initialization.random_effect_indices.len();
        let mut subject_log_acceptance_sums = vec![0.0; self.initialization.subject_ids.len()];
        let mut subject_proposal_counts = vec![0usize; self.initialization.subject_ids.len()];
        let mut parameter_accept_counts = vec![0usize; n_parameters];
        let mut parameter_proposal_counts = vec![0usize; n_parameters];
        let mut kernel_counts = [KernelCounters::default(); 4];
        let subset_step_sizes_before = self.saemix_subset_step_sizes.clone();
        let mut saemix_component_step_sizes_after = None;

        if let Some(policy) = self.saemix_mcmc {
            let lower = cholesky_lower(&self.omega)?;
            for _ in 0..policy.iterations[0] {
                for subject_index in 0..self.initialization.subject_ids.len() {
                    for chain_index in 0..self.initialization.n_chains {
                        let current_eta = self.etas[subject_index][chain_index].clone();
                        let current_score = self.score_subject_latents(
                            subject_index,
                            &current_eta,
                            &self.kappas[subject_index][chain_index],
                        )?;
                        let proposed_eta = self.prior_independence_eta(&lower)?;
                        let proposed_score = self.score_subject_latents(
                            subject_index,
                            &proposed_eta,
                            &self.kappas[subject_index][chain_index],
                        )?;
                        let log_acceptance_ratio =
                            saemix_prior_independence_log_acceptance(current_score, proposed_score);
                        subject_log_acceptance_sums[subject_index] += log_acceptance_ratio;
                        subject_proposal_counts[subject_index] += 1;
                        kernel_counts[0].proposals += 1;
                        eta_proposed += 1;
                        if !log_acceptance_ratio.is_finite() {
                            kernel_counts[0].non_finite += 1;
                            eta_non_finite += 1;
                        }
                        if self.accept_proposal(log_acceptance_ratio) {
                            self.etas[subject_index][chain_index] = proposed_eta;
                            kernel_counts[0].accepted += 1;
                            eta_accepted += 1;
                        } else {
                            kernel_counts[0].rejected += 1;
                            eta_rejected += 1;
                        }
                    }
                }
            }
        }

        // Compound-kernel order: Omega-scaled eta blocks first, followed by
        // component eta walks and occasion-level kappa blocks. Eta blocks are
        // opt-in.
        for _ in 0..self.eta_block_iterations {
            for subject_index in 0..self.initialization.subject_ids.len() {
                for chain_index in 0..self.initialization.n_chains {
                    let current_eta = self.etas[subject_index][chain_index].clone();
                    let proposed_eta = self.block_random_walk_eta(&current_eta, subject_index)?;
                    let log_acceptance_ratio = self.proposal_log_acceptance_ratio(
                        subject_index,
                        chain_index,
                        &proposed_eta,
                    )?;
                    subject_log_acceptance_sums[subject_index] += log_acceptance_ratio;
                    subject_proposal_counts[subject_index] += 1;
                    eta_block_subject_proposal_counts[subject_index] += 1;
                    self.eta_block_adaptation_proposal_counts[subject_index] += 1;
                    eta_block_proposed += 1;
                    eta_proposed += 1;
                    if !log_acceptance_ratio.is_finite() {
                        eta_block_non_finite += 1;
                        eta_non_finite += 1;
                    }
                    if self.accept_proposal(log_acceptance_ratio) {
                        self.etas[subject_index][chain_index] = proposed_eta;
                        eta_block_subject_accept_counts[subject_index] += 1;
                        self.eta_block_adaptation_accept_counts[subject_index] += 1;
                        eta_block_accepted += 1;
                        eta_accepted += 1;
                    } else {
                        eta_block_rejected += 1;
                        eta_rejected += 1;
                    }
                }
            }
        }

        let component_iterations = self
            .saemix_mcmc
            .map_or(self.mcmc_iterations, |policy| policy.iterations[1]);
        for _ in 0..component_iterations {
            for subject_index in 0..self.initialization.subject_ids.len() {
                for chain_index in 0..self.initialization.n_chains {
                    for parameter_index in 0..n_parameters {
                        let current_eta = self.etas[subject_index][chain_index].clone();
                        let proposed_eta =
                            self.component_random_walk_eta(&current_eta, parameter_index);
                        let log_acceptance_ratio = self.proposal_log_acceptance_ratio(
                            subject_index,
                            chain_index,
                            &proposed_eta,
                        )?;
                        subject_log_acceptance_sums[subject_index] += log_acceptance_ratio;
                        subject_proposal_counts[subject_index] += 1;
                        parameter_proposal_counts[parameter_index] += 1;
                        eta_proposed += 1;
                        if self.saemix_mcmc.is_some() {
                            kernel_counts[1].proposals += 1;
                        }
                        if !log_acceptance_ratio.is_finite() {
                            eta_non_finite += 1;
                            if self.saemix_mcmc.is_some() {
                                kernel_counts[1].non_finite += 1;
                            }
                        }
                        if self.accept_proposal(log_acceptance_ratio) {
                            self.etas[subject_index][chain_index] = proposed_eta;
                            parameter_accept_counts[parameter_index] += 1;
                            eta_accepted += 1;
                            if self.saemix_mcmc.is_some() {
                                kernel_counts[1].accepted += 1;
                            }
                        } else {
                            eta_rejected += 1;
                            if self.saemix_mcmc.is_some() {
                                kernel_counts[1].rejected += 1;
                            }
                        }
                    }

                    // Gibbs sweep over occasion-specific κ blocks. Every
                    // proposal is evaluated against the full subject posterior,
                    // keeping η and all other occasions fixed.
                    if self.omega_iov.is_some() {
                        for occasion_index in 0..self.kappas[subject_index][chain_index].len() {
                            let current_kappa =
                                self.kappas[subject_index][chain_index][occasion_index].clone();
                            let proposed_kappa =
                                self.block_random_walk_kappa(&current_kappa, subject_index)?;
                            let log_acceptance_ratio = self.kappa_proposal_log_acceptance_ratio(
                                subject_index,
                                chain_index,
                                occasion_index,
                                &proposed_kappa,
                            )?;
                            subject_log_acceptance_sums[subject_index] += log_acceptance_ratio;
                            subject_proposal_counts[subject_index] += 1;
                            kappa_proposed += 1;
                            kappa_subject_proposal_counts[subject_index] += 1;
                            self.kappa_adaptation_proposal_counts[subject_index] += 1;
                            if !log_acceptance_ratio.is_finite() {
                                kappa_non_finite += 1;
                            }
                            if self.accept_proposal(log_acceptance_ratio) {
                                self.kappas[subject_index][chain_index][occasion_index] =
                                    proposed_kappa;
                                kappa_accepted += 1;
                                kappa_subject_accept_counts[subject_index] += 1;
                                self.kappa_adaptation_accept_counts[subject_index] += 1;
                            } else {
                                kappa_rejected += 1;
                            }
                        }
                    }
                }
            }
        }

        if let Some(policy) = self.saemix_mcmc {
            for parameter_index in 0..n_parameters {
                let proposed = parameter_proposal_counts[parameter_index];
                if proposed > 0 {
                    let acceptance =
                        parameter_accept_counts[parameter_index] as f64 / proposed as f64;
                    self.proposal_step_sizes[parameter_index] = saemix_adapt_step_size(
                        self.proposal_step_sizes[parameter_index],
                        acceptance,
                        policy,
                    )?;
                }
            }
            saemix_component_step_sizes_after = Some(self.proposal_step_sizes.clone());

            let mut subset_accept_counts = vec![0usize; n_parameters];
            let mut subset_proposal_counts = vec![0usize; n_parameters];
            let mut active_subset_size = None;
            for _ in 0..policy.iterations[2] {
                let (subset_size, groups) = self.saemix_subset_groups(n_parameters);
                active_subset_size = Some(subset_size);
                for group in groups {
                    for subject_index in 0..self.initialization.subject_ids.len() {
                        for chain_index in 0..self.initialization.n_chains {
                            let current_eta = self.etas[subject_index][chain_index].clone();
                            let proposed_eta =
                                self.subset_random_walk_eta(&current_eta, &group, subset_size);
                            let log_acceptance_ratio = self.proposal_log_acceptance_ratio(
                                subject_index,
                                chain_index,
                                &proposed_eta,
                            )?;
                            subject_log_acceptance_sums[subject_index] += log_acceptance_ratio;
                            subject_proposal_counts[subject_index] += 1;
                            kernel_counts[2].proposals += 1;
                            eta_proposed += 1;
                            for parameter in &group {
                                subset_proposal_counts[*parameter] += 1;
                            }
                            if !log_acceptance_ratio.is_finite() {
                                kernel_counts[2].non_finite += 1;
                                eta_non_finite += 1;
                            }
                            if self.accept_proposal(log_acceptance_ratio) {
                                self.etas[subject_index][chain_index] = proposed_eta;
                                kernel_counts[2].accepted += 1;
                                eta_accepted += 1;
                                for parameter in &group {
                                    subset_accept_counts[*parameter] += 1;
                                }
                            } else {
                                kernel_counts[2].rejected += 1;
                                eta_rejected += 1;
                            }
                        }
                    }
                }
            }
            if let Some(subset_size) = active_subset_size {
                for parameter_index in 0..n_parameters {
                    let proposed = subset_proposal_counts[parameter_index];
                    if proposed > 0 {
                        let acceptance =
                            subset_accept_counts[parameter_index] as f64 / proposed as f64;
                        if n_parameters == 1 {
                            self.proposal_step_sizes[0] = saemix_adapt_step_size(
                                self.proposal_step_sizes[0],
                                acceptance,
                                policy,
                            )?;
                            self.saemix_subset_step_sizes[0][0] = self.proposal_step_sizes[0];
                        } else {
                            let current =
                                self.saemix_subset_step_sizes[parameter_index][subset_size - 1];
                            self.saemix_subset_step_sizes[parameter_index][subset_size - 1] =
                                saemix_adapt_step_size(current, acceptance, policy)?;
                        }
                    }
                }
            }

            if policy.iterations[3] > 0 && self.cycle < policy.map_cycles {
                let mut distributions = Vec::with_capacity(self.initialization.subject_ids.len());
                for subject_index in 0..self.initialization.subject_ids.len() {
                    let distribution = self.saemix_map_distribution(subject_index, policy)?;
                    for chain in &mut self.etas[subject_index] {
                        *chain = distribution.mode.clone();
                    }
                    distributions.push(distribution);
                }
                for _ in 0..policy.iterations[3] {
                    for (subject_index, distribution) in distributions.iter().enumerate() {
                        let mode = &distribution.mode;
                        let covariance = &distribution.covariance;
                        let lower = &distribution.lower;
                        for chain_index in 0..self.initialization.n_chains {
                            let current_eta = self.etas[subject_index][chain_index].clone();
                            let standard_normals = (0..n_parameters)
                                .map(|_| self.standard_normal())
                                .collect::<Vec<_>>();
                            let proposed_eta =
                                correlated_random_walk(mode, lower, &standard_normals, 1.0)?;
                            let current_score = self.score_subject_latents(
                                subject_index,
                                &current_eta,
                                &self.kappas[subject_index][chain_index],
                            )?;
                            let proposed_score = self.score_subject_latents(
                                subject_index,
                                &proposed_eta,
                                &self.kappas[subject_index][chain_index],
                            )?;
                            let current_centered = current_eta
                                .iter()
                                .zip(mode)
                                .map(|(value, center)| value - center)
                                .collect::<Vec<_>>();
                            let proposed_centered = proposed_eta
                                .iter()
                                .zip(mode)
                                .map(|(value, center)| value - center)
                                .collect::<Vec<_>>();
                            let log_acceptance_ratio = saemix_map_independence_log_acceptance(
                                current_score,
                                proposed_score,
                                &current_centered,
                                &proposed_centered,
                                covariance,
                            )?;
                            subject_log_acceptance_sums[subject_index] += log_acceptance_ratio;
                            subject_proposal_counts[subject_index] += 1;
                            kernel_counts[3].proposals += 1;
                            eta_proposed += 1;
                            if !log_acceptance_ratio.is_finite() {
                                kernel_counts[3].non_finite += 1;
                                eta_non_finite += 1;
                            }
                            if self.accept_proposal(log_acceptance_ratio) {
                                self.etas[subject_index][chain_index] = proposed_eta;
                                kernel_counts[3].accepted += 1;
                                eta_accepted += 1;
                            } else {
                                kernel_counts[3].rejected += 1;
                                eta_rejected += 1;
                            }
                        }
                    }
                }
            }
        }

        self.refresh_subject_scores_from_chains()?;
        self.last_log_acceptance_ratios = subject_log_acceptance_sums
            .into_iter()
            .zip(subject_proposal_counts)
            .map(|(sum, count)| if count > 0 { sum / count as f64 } else { 0.0 })
            .collect();
        let proposed = eta_proposed + kappa_proposed;
        let accepted = eta_accepted + kappa_accepted;
        self.last_acceptance_rate = if proposed > 0 {
            Some(accepted as f64 / proposed as f64)
        } else {
            None
        };
        self.last_eta_block_acceptance_rate = if self.eta_block_iterations > 0 {
            Some(eta_block_accepted as f64 / eta_block_proposed.max(1) as f64)
        } else {
            None
        };
        self.last_kappa_acceptance_rate = if self.omega_iov.is_some() {
            Some(kappa_accepted as f64 / kappa_proposed.max(1) as f64)
        } else {
            None
        };
        self.last_rejected_proposals = Some(eta_rejected + kappa_rejected);
        self.last_non_finite_proposals = Some(eta_non_finite + kappa_non_finite);
        self.last_parameter_acceptance_rates = parameter_accept_counts
            .iter()
            .zip(parameter_proposal_counts.iter())
            .map(|(accepted, proposed)| {
                if *proposed > 0 {
                    *accepted as f64 / *proposed as f64
                } else {
                    0.0
                }
            })
            .collect();
        if self.saemix_mcmc.is_none() {
            for parameter_index in 0..n_parameters {
                self.adaptation_accept_counts[parameter_index] +=
                    parameter_accept_counts[parameter_index];
                self.adaptation_proposal_counts[parameter_index] +=
                    parameter_proposal_counts[parameter_index];
            }
            self.steps_since_adapt += 1;
            self.adapt_proposal_step_sizes();
        }
        let mcmc_kernel_diagnostics = if self.saemix_mcmc.is_some() {
            let kernels = [
                SaemMcmcKernel::PriorIndependence,
                SaemMcmcKernel::ComponentRandomWalk,
                SaemMcmcKernel::RotatingSubset,
                SaemMcmcKernel::MapIndependence,
            ];
            kernels
                .into_iter()
                .enumerate()
                .map(|(index, kernel)| {
                    let (before, after) = match kernel {
                        SaemMcmcKernel::ComponentRandomWalk => (
                            vec![eta_step_sizes_before.clone()],
                            vec![saemix_component_step_sizes_after
                                .clone()
                                .unwrap_or_else(|| self.proposal_step_sizes.clone())],
                        ),
                        SaemMcmcKernel::RotatingSubset => (
                            subset_step_sizes_before.clone(),
                            self.saemix_subset_step_sizes.clone(),
                        ),
                        _ => (Vec::new(), Vec::new()),
                    };
                    SaemMcmcKernelDiagnostics {
                        kernel,
                        proposals: kernel_counts[index].proposals,
                        accepted: kernel_counts[index].accepted,
                        rejected: kernel_counts[index].rejected,
                        non_finite: kernel_counts[index].non_finite,
                        proposal_scales_before: before,
                        proposal_scales_after: after,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        let phase = self.initialization.schedule.phase(self.cycle);
        let omega_update = pending_covariance_update_diagnostics(
            phase,
            true,
            self.initialization.omega.has_estimated_entries(),
        );
        let omega_iov_update = pending_covariance_update_diagnostics(
            phase,
            self.initialization.omega_iov.is_some(),
            self.initialization
                .omega_iov
                .as_ref()
                .is_some_and(ResolvedOmega::has_estimated_entries),
        );
        self.cycle_diagnostics.push(SaemCycleDiagnostics {
            iteration: self.cycle,
            phase,
            stochastic_approximation_step: self
                .initialization
                .schedule
                .stochastic_approximation_step(self.cycle),
            covariance_step: self.initialization.schedule.covariance_step(self.cycle),
            mcmc_kernel_diagnostics,
            eta_proposals: eta_proposed,
            eta_accepted,
            eta_rejected,
            eta_non_finite,
            eta_parameter_acceptance_rates: self.last_parameter_acceptance_rates.clone(),
            eta_proposal_step_sizes_before_adaptation: eta_step_sizes_before,
            eta_proposal_step_sizes_after_adaptation: self.proposal_step_sizes.clone(),
            eta_block_proposals: eta_block_proposed,
            eta_block_accepted,
            eta_block_rejected,
            eta_block_non_finite,
            eta_block_subject_acceptance_rates: eta_block_subject_accept_counts
                .iter()
                .zip(eta_block_subject_proposal_counts.iter())
                .map(|(accepted, proposed)| {
                    if *proposed > 0 {
                        *accepted as f64 / *proposed as f64
                    } else {
                        0.0
                    }
                })
                .collect(),
            eta_block_step_sizes_before_adaptation: eta_block_step_sizes_before,
            eta_block_step_sizes_after_adaptation: self.eta_block_step_sizes.clone(),
            kappa_proposals: kappa_proposed,
            kappa_accepted,
            kappa_rejected,
            kappa_non_finite,
            kappa_subject_acceptance_rates: kappa_subject_accept_counts
                .iter()
                .zip(kappa_subject_proposal_counts.iter())
                .map(|(accepted, proposed)| {
                    if *proposed > 0 {
                        *accepted as f64 / *proposed as f64
                    } else {
                        0.0
                    }
                })
                .collect(),
            kappa_proposal_step_sizes_before_adaptation: kappa_step_sizes_before,
            kappa_proposal_step_sizes_after_adaptation: self.kappa_proposal_step_sizes.clone(),
            simulated_annealing_active: self.cycle
                <= self.initialization.schedule.variance_floor_iterations,
            population_parameters: self.population_parameters.clone(),
            omega: self.omega.clone(),
            omega_iov: self.omega_iov.clone(),
            residual_error_estimates: self.residual_error_estimates(),
            residual_diagnostics: Vec::new(),
            conditional_negative_log_likelihood: self.negative_log_likelihood,
            eta_log_prior: self.subject_log_priors.iter().sum(),
            kappa_log_prior: self.subject_kappa_log_priors.iter().sum(),
            omega_update_rejected: false,
            omega_iov_update_rejected: false,
            omega_update,
            omega_iov_update,
            omega_relative_spd_margin: None,
            omega_iov_relative_spd_margin: None,
            covariate_betas: self.covariate_model.as_ref().map(|model| {
                model
                    .estimates()
                    .iter()
                    .map(|estimate| estimate.estimate())
                    .collect()
            }),
            covariate_beta_estimated: self.covariate_model.as_ref().map(|model| {
                model
                    .estimates()
                    .iter()
                    .map(|estimate| estimate.estimated())
                    .collect()
            }),
        });
        self.negative_log_likelihood = negative_log_likelihood(&self.subject_log_likelihoods);
        Ok(())
    }

    fn m_step(&mut self) -> Result<()> {
        let parameter_step = self
            .initialization
            .schedule
            .stochastic_approximation_step(self.cycle);
        let covariance_step = self.initialization.schedule.covariance_step(self.cycle);
        if self.covariate_model.is_some() {
            let observed = self.current_covariate_statistics()?;
            self.covariate_statistics
                .as_mut()
                .expect("covariate model has initialized statistics")
                .stochastic_update(&observed, parameter_step)?;
        } else {
            let observed_statistics = self.current_phi_statistics()?;
            self.sufficient_statistics.stochastic_update_with_steps(
                &observed_statistics,
                parameter_step,
                covariance_step,
            )?;
        }

        if let Some(second_moment) = self.iov_second_moment.as_mut() {
            let observed_second_moment = covariance_from_kappas(&self.kappas)?;
            *second_moment =
                &*second_moment + &((&observed_second_moment - &*second_moment) * covariance_step);
        }

        // Pure burn-in warms the latent chains and their centered covariance
        // statistics while theta, Omega, Omega_IOV, and sigma remain fixed. Raw
        // covariate phi moments remain unchanged, matching their zero SA gain.
        if parameter_step == 0.0 {
            let observed_second_moment = second_moment_from_etas(&self.etas)?;
            self.iiv_second_moment = &self.iiv_second_moment
                + &((&observed_second_moment - &self.iiv_second_moment) * covariance_step);
            self.finalize_cycle_diagnostics()?;
            return Ok(());
        }

        let pre_update_residual_evidence = self.current_residual_statistics_and_information()?;
        if self.covariate_model.is_some() {
            // The raw first and second phi moments already share the SAEM gain.
            // Keep their centered covariance candidate coherent; exploration
            // robustness is applied later to the accepted Omega iterate rather
            // than introducing a second sufficient-statistic recursion.
            self.iiv_second_moment = self.update_covariate_population_and_recenter_etas()?;
        } else {
            self.update_population_and_recenter_etas()?;
            let observed_second_moment = second_moment_from_etas(&self.etas)?;
            self.iiv_second_moment = &self.iiv_second_moment
                + &((&observed_second_moment - &self.iiv_second_moment) * covariance_step);
        }

        self.update_non_iiv_population(parameter_step)?;
        let (observed_residual_statistics, information_replicates) = pre_update_residual_evidence;
        match information_replicates {
            Ok(replicates) => self.information.update(&replicates, parameter_step),
            Err(reason) => self
                .information
                .mark_unavailable(information_failure_status(reason)),
        }
        let mut residual_diagnostics = self
            .error_models
            .models()
            .iter()
            .map(|(output_index, _)| {
                let statistic = observed_residual_statistics
                    .output(output_index)
                    .unwrap_or_default();
                ResidualCycleDiagnostics {
                    output: self
                        .error_models
                        .output_name(output_index)
                        .map(str::to_owned)
                        .unwrap_or_else(|| format!("output_{output_index}")),
                    output_index,
                    prediction_evaluation_count: statistic.observation_count,
                    proportional_floor_count: statistic.proportional_floor_count,
                    non_finite_prediction_count: statistic.non_finite_prediction_count,
                    exponential_domain_violation_count: statistic
                        .exponential_domain_violation_count,
                    update_rejected: false,
                    optimizer_objective: None,
                    optimizer_converged: None,
                    optimizer_iterations: None,
                    optimizer_termination: None,
                    combined_additive_collapse_warning: false,
                }
            })
            .collect::<Vec<_>>();
        let residual_observations = (0..self.error_models.len())
            .map(|output_index| {
                observed_residual_statistics
                    .observations(output_index)
                    .unwrap_or_default()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        self.residual_statistics = self
            .residual_statistics
            .stochastic_update(observed_residual_statistics, parameter_step);

        if self
            .initialization
            .schedule
            .covariance_update_active(self.cycle)
        {
            if self.initialization.omega.has_estimated_entries() {
                let phase = self.initialization.schedule.phase(self.cycle);
                let update = if self.covariate_model.is_some() && phase == SaemPhase::Exploration {
                    self.initialization
                        .omega
                        .update_with_status_and_max_fraction(
                            &self.omega,
                            &self.iiv_second_moment,
                            self.initialization.schedule.minimum_variance,
                            covariate_omega_update_maximum_fraction(true, phase, covariance_step),
                        )?
                } else {
                    // Preserve the established floor-after-interpolation path
                    // for non-covariate IIV and for uncapped covariate smoothing.
                    self.initialization.omega.update_with_status(
                        &self.omega,
                        &self.iiv_second_moment,
                        self.initialization.schedule.minimum_variance,
                    )?
                };
                let status = update.status;
                let update_diagnostics =
                    completed_covariance_update_diagnostics(&self.iiv_second_moment, &update)?;
                self.omega = update.matrix;
                if let Some(diagnostics) = self.cycle_diagnostics.last_mut() {
                    diagnostics.omega_update_rejected = status == CovarianceUpdateStatus::Rejected;
                    diagnostics.omega_update = update_diagnostics;
                }
            }
            if let (Some(specification), Some(omega_iov), Some(second_moment)) = (
                self.initialization.omega_iov.as_ref(),
                self.omega_iov.as_mut(),
                self.iov_second_moment.as_ref(),
            ) {
                if specification.has_estimated_entries() {
                    let update = specification.update_with_status(
                        omega_iov,
                        second_moment,
                        self.initialization.schedule.minimum_iov_variance,
                    )?;
                    let status = update.status;
                    let update_diagnostics =
                        completed_covariance_update_diagnostics(second_moment, &update)?;
                    *omega_iov = update.matrix;
                    if let Some(diagnostics) = self.cycle_diagnostics.last_mut() {
                        diagnostics.omega_iov_update_rejected =
                            status == CovarianceUpdateStatus::Rejected;
                        diagnostics.omega_iov_update = update_diagnostics;
                    }
                }
            }
        }
        for residual_diagnostic in &mut residual_diagnostics {
            let outeq = residual_diagnostic.output_index;
            if !self.error_models.is_estimated(outeq) {
                continue;
            }
            let Some(model) = self.error_models.models().get(outeq).copied() else {
                residual_diagnostic.update_rejected = true;
                continue;
            };
            if let ResidualErrorModel::Combined { a, b } = model {
                match optimize_combined_residual(
                    &residual_observations[outeq],
                    a,
                    b,
                    self.error_models.combined_component_estimated(outeq),
                    self.initialization.schedule.minimum_residual_sigma,
                    self.residual_optimizer_max_iterations as u64,
                ) {
                    Ok(solution) => {
                        let component_estimated =
                            self.error_models.combined_component_estimated(outeq);
                        let additive_sd = applied_combined_residual_component(
                            &self.initialization.schedule,
                            self.cycle,
                            a,
                            solution.additive_sd,
                            component_estimated[0],
                        );
                        let proportional_sd = applied_combined_residual_component(
                            &self.initialization.schedule,
                            self.cycle,
                            b,
                            solution.proportional_sd,
                            component_estimated[1],
                        );
                        residual_diagnostic.combined_additive_collapse_warning =
                            combined_additive_sigma_collapsed(additive_sd, component_estimated[0]);
                        update_estimated_combined_residual_model(
                            &mut self.error_models,
                            outeq,
                            additive_sd,
                            proportional_sd,
                        );
                        residual_diagnostic.optimizer_objective = Some(solution.objective);
                        residual_diagnostic.optimizer_converged = Some(solution.converged);
                        residual_diagnostic.optimizer_iterations = Some(solution.iterations);
                        residual_diagnostic.optimizer_termination = Some(solution.termination);
                    }
                    Err(error) => {
                        residual_diagnostic.update_rejected = true;
                        residual_diagnostic.optimizer_termination = Some(error.to_string());
                    }
                }
                continue;
            }
            if let ResidualErrorModel::CorrelatedCombined { a, b, rho } = model {
                match optimize_correlated_combined_residual(
                    &residual_observations[outeq],
                    a,
                    b,
                    rho,
                    self.error_models
                        .correlated_combined_component_estimated(outeq),
                    self.initialization.schedule.minimum_residual_sigma,
                    self.residual_optimizer_max_iterations as u64,
                ) {
                    Ok(solution) => {
                        let component_estimated = self
                            .error_models
                            .correlated_combined_component_estimated(outeq);
                        let additive_sd = applied_combined_residual_component(
                            &self.initialization.schedule,
                            self.cycle,
                            a,
                            solution.additive_sd,
                            component_estimated[0],
                        );
                        let proportional_sd = applied_combined_residual_component(
                            &self.initialization.schedule,
                            self.cycle,
                            b,
                            solution.proportional_sd,
                            component_estimated[1],
                        );
                        let correlation = applied_correlated_residual_correlation(
                            &self.initialization.schedule,
                            self.cycle,
                            rho,
                            solution.correlation,
                            component_estimated[2],
                        );
                        if !correlation.is_finite() || correlation <= -1.0 || correlation >= 1.0 {
                            residual_diagnostic.update_rejected = true;
                            residual_diagnostic.optimizer_termination = Some(
                                "correlated-combined residual update left (-1, 1)".to_string(),
                            );
                            continue;
                        }
                        residual_diagnostic.combined_additive_collapse_warning =
                            combined_additive_sigma_collapsed(additive_sd, component_estimated[0]);
                        update_estimated_correlated_combined_residual_model(
                            &mut self.error_models,
                            outeq,
                            additive_sd,
                            proportional_sd,
                            correlation,
                        );
                        residual_diagnostic.optimizer_objective = Some(solution.objective);
                        residual_diagnostic.optimizer_converged = Some(solution.converged);
                        residual_diagnostic.optimizer_iterations = Some(solution.iterations);
                        residual_diagnostic.optimizer_termination = Some(solution.termination);
                    }
                    Err(error) => {
                        residual_diagnostic.update_rejected = true;
                        residual_diagnostic.optimizer_termination = Some(error.to_string());
                    }
                }
                continue;
            }
            let Some(candidate_sigma) = self
                .residual_statistics
                .output(outeq)
                .and_then(|statistic| statistic.sigma())
            else {
                residual_diagnostic.update_rejected = true;
                continue;
            };
            let previous_sigma = primary_sigma_parameter(&model);
            let sigma = self.initialization.schedule.guarded_residual_sigma(
                self.cycle,
                previous_sigma,
                candidate_sigma,
            );
            update_estimated_simple_residual_model_with_sigma(&mut self.error_models, outeq, sigma);
        }
        if let Some(diagnostics) = self.cycle_diagnostics.last_mut() {
            diagnostics.residual_diagnostics = residual_diagnostics;
        }
        self.residual_sigmas = primary_sigma_parameters(self.error_models.models());
        self.refresh_subject_scores_from_chains()?;
        self.negative_log_likelihood = negative_log_likelihood(&self.subject_log_likelihoods);
        self.update_iterate_average()?;
        self.finalize_cycle_diagnostics()?;
        Ok(())
    }

    fn update_iterate_average(&mut self) -> Result<()> {
        if self.initialization.schedule.phase(self.cycle) != SaemPhase::Smoothing
            || !matches!(
                self.config.estimator_policy,
                SaemEstimatorPolicy::AveragedIterates { .. }
            )
        {
            return Ok(());
        }
        let population_phi = population_phi(
            &self.population_parameters,
            &self.initialization.parameter_scales,
        )?;
        let residual_models = self
            .error_models
            .models()
            .iter()
            .map(|(output_index, model)| (output_index, *model))
            .collect::<Vec<_>>();
        let residual_model_width = self.error_models.models().len();
        let Some(average) = self.iterate_average.as_mut() else {
            self.iterate_average = Some(SaemIterateAverage {
                population_phi,
                covariate_betas: self.covariate_model.as_ref().map(|model| {
                    model
                        .estimates()
                        .iter()
                        .map(|estimate| estimate.estimate())
                        .collect()
                }),
                omega: self.omega.clone(),
                omega_iov: self.omega_iov.clone(),
                residual_model_width,
                residual_models,
                start_cycle: self.cycle,
                count: 1,
            });
            return Ok(());
        };
        let next_count = average.count + 1;
        for (index, value) in population_phi.iter().copied().enumerate() {
            if self.initialization.estimated_parameters[index] {
                average.population_phi[index] =
                    incremental_average(average.population_phi[index], value, next_count);
            }
        }
        if let (Some(average_betas), Some(model)) = (
            average.covariate_betas.as_mut(),
            self.covariate_model.as_ref(),
        ) {
            for (index, estimate) in model.estimates().iter().enumerate() {
                if estimate.estimated() {
                    average_betas[index] =
                        incremental_average(average_betas[index], estimate.estimate(), next_count);
                }
            }
        }
        average_covariance(
            &mut average.omega,
            &self.omega,
            self.initialization.omega.estimated_mask(),
            next_count,
        );
        if let (Some(average_iov), Some(current_iov), Some(specification)) = (
            average.omega_iov.as_mut(),
            self.omega_iov.as_ref(),
            self.initialization.omega_iov.as_ref(),
        ) {
            average_covariance(
                average_iov,
                current_iov,
                specification.estimated_mask(),
                next_count,
            );
        }
        if residual_model_width != average.residual_model_width
            || residual_models.len() != average.residual_models.len()
        {
            anyhow::bail!("residual output declarations changed while accumulating SAEM averages");
        }
        for ((average_output_index, previous), (output_index, current)) in
            average.residual_models.iter_mut().zip(residual_models)
        {
            if *average_output_index != output_index {
                anyhow::bail!(
                    "residual output declarations changed while accumulating SAEM averages"
                );
            }
            let estimated = self.error_models.is_estimated(output_index);
            let components = self.error_models.combined_component_estimated(output_index);
            let correlated_components = self
                .error_models
                .correlated_combined_component_estimated(output_index);
            *previous = average_residual_model(
                *previous,
                current,
                estimated,
                components,
                correlated_components,
                next_count,
            )?;
        }
        average.count = next_count;
        Ok(())
    }

    fn install_iterate_average(&mut self) -> Result<SaemEstimatorMetadata> {
        let policy = self.config.estimator_policy;
        let Some(average) = self.iterate_average.clone() else {
            tracing::info!("averaged SAEM estimate was not available; retaining terminal iterate");
            return Ok(SaemEstimatorMetadata {
                policy,
                ..SaemEstimatorMetadata::default()
            });
        };
        let terminal_phi = population_phi(
            &self.population_parameters,
            &self.initialization.parameter_scales,
        )?;
        validate_average_population(&average.population_phi, &self.initialization)?;
        validate_average_covariance(&average.omega, &self.initialization.omega, "Omega")?;
        if let (Some(matrix), Some(specification)) = (
            average.omega_iov.as_ref(),
            self.initialization.omega_iov.as_ref(),
        ) {
            validate_average_covariance(matrix, specification, "Omega_IOV")?;
        }
        validate_average_residuals(
            average.residual_model_width,
            &average.residual_models,
            &self.error_models,
        )?;

        self.population_parameters = population_psi(
            &average.population_phi,
            &self.initialization.parameter_scales,
        )?;
        if let (Some(model), Some(beta_values), Some(old_means)) = (
            self.covariate_model.as_ref(),
            average.covariate_betas.as_ref(),
            self.subject_mu_phi.as_ref(),
        ) {
            let averaged_model = model.with_estimates(beta_values)?;
            let new_rows = averaged_model.subject_population_parameters(
                &average.population_phi,
                &self.initialization.parameter_scales,
            )?;
            let new_means = new_rows
                .iter()
                .map(|row| row.phi().to_vec())
                .collect::<Vec<_>>();
            for (subject_index, chains) in self.etas.iter_mut().enumerate() {
                let old_random = self
                    .initialization
                    .random_effect_indices
                    .iter()
                    .map(|index| old_means[subject_index][*index])
                    .collect::<Vec<_>>();
                let new_random = self
                    .initialization
                    .random_effect_indices
                    .iter()
                    .map(|index| new_means[subject_index][*index])
                    .collect::<Vec<_>>();
                for eta in chains {
                    rebase_eta(eta, &old_random, &new_random)?;
                }
            }
            self.covariate_model = Some(averaged_model);
            self.subject_mu_phi = Some(new_means);
        } else {
            for (eta_index, parameter_index) in self
                .initialization
                .random_effect_indices
                .iter()
                .copied()
                .enumerate()
            {
                let shift = terminal_phi[parameter_index] - average.population_phi[parameter_index];
                for subject_chains in &mut self.etas {
                    for eta in subject_chains {
                        eta[eta_index] += shift;
                    }
                }
            }
        }
        self.omega = average.omega;
        self.omega_iov = average.omega_iov;
        for (output_index, model) in average.residual_models {
            match model {
                ResidualErrorModel::Combined { a, b } => update_estimated_combined_residual_model(
                    &mut self.error_models,
                    output_index,
                    a,
                    b,
                ),
                ResidualErrorModel::CorrelatedCombined { a, b, rho } => {
                    update_estimated_correlated_combined_residual_model(
                        &mut self.error_models,
                        output_index,
                        a,
                        b,
                        rho,
                    )
                }
                ResidualErrorModel::Constant { .. }
                | ResidualErrorModel::Proportional { .. }
                | ResidualErrorModel::Exponential { .. } => {
                    update_estimated_simple_residual_model_with_sigma(
                        &mut self.error_models,
                        output_index,
                        primary_sigma_parameter(&model),
                    )
                }
            }
        }
        self.residual_sigmas = primary_sigma_parameters(self.error_models.models());
        self.refresh_subject_scores_from_chains()?;
        self.negative_log_likelihood = negative_log_likelihood(&self.subject_log_likelihoods);
        tracing::info!(
            start_cycle = average.start_cycle,
            averaged_iterations = average.count,
            "installed averaged SAEM estimate"
        );
        Ok(SaemEstimatorMetadata {
            policy,
            average_applied: true,
            averaging_start_cycle: Some(average.start_cycle),
            averaged_iterations: average.count,
        })
    }

    fn residual_error_estimates(&self) -> Vec<ResidualErrorEstimate> {
        self.error_models
            .models()
            .iter()
            .map(|(output_index, model)| {
                let model = *model;
                let combined_components =
                    self.error_models.combined_component_estimated(output_index);
                let correlated_components = self
                    .error_models
                    .correlated_combined_component_estimated(output_index);
                let is_combined = matches!(model, ResidualErrorModel::Combined { .. });
                let is_correlated = matches!(model, ResidualErrorModel::CorrelatedCombined { .. });
                ResidualErrorEstimate {
                    output: self
                        .error_models
                        .output_name(output_index)
                        .map(str::to_owned)
                        .expect("declared residual models have output names"),
                    output_index,
                    model,
                    estimated: self.error_models.is_estimated(output_index),
                    combined_additive_estimated: if is_combined {
                        Some(combined_components[0])
                    } else {
                        is_correlated.then_some(correlated_components[0])
                    },
                    combined_proportional_estimated: if is_combined {
                        Some(combined_components[1])
                    } else {
                        is_correlated.then_some(correlated_components[1])
                    },
                    correlation_estimated: is_correlated.then_some(correlated_components[2]),
                }
            })
            .collect()
    }

    fn finalize_cycle_diagnostics(&mut self) -> Result<()> {
        let population_parameters = self.population_parameters.clone();
        let omega = self.omega.clone();
        let omega_iov = self.omega_iov.clone();
        let residual_error_estimates = self.residual_error_estimates();
        let conditional_negative_log_likelihood = self.negative_log_likelihood;
        let eta_log_prior = self.subject_log_priors.iter().sum();
        let kappa_log_prior = self.subject_kappa_log_priors.iter().sum();
        let (omega_relative_spd_margin, omega_iov_relative_spd_margin) =
            if self.config.covariance_stability.is_some() {
                let initial_omega = self.initialization.omega.initial();
                let omega_margin = (initial_omega.nrows() > 0)
                    .then(|| relative_spd_margin(&omega, initial_omega))
                    .transpose()?;
                let omega_iov_margin =
                    match (self.initialization.omega_iov.as_ref(), omega_iov.as_ref()) {
                        (Some(specification), Some(matrix))
                            if specification.initial().nrows() > 0 =>
                        {
                            Some(relative_spd_margin(matrix, specification.initial())?)
                        }
                        _ => None,
                    };
                (omega_margin, omega_iov_margin)
            } else {
                (None, None)
            };
        let covariate_betas = self.covariate_model.as_ref().map(|model| {
            model
                .estimates()
                .iter()
                .map(|estimate| estimate.estimate())
                .collect()
        });
        let covariate_beta_estimated = self.covariate_model.as_ref().map(|model| {
            model
                .estimates()
                .iter()
                .map(|estimate| estimate.estimated())
                .collect()
        });
        if let Some(diagnostics) = self.cycle_diagnostics.last_mut() {
            diagnostics.population_parameters = population_parameters;
            diagnostics.omega = omega;
            diagnostics.omega_iov = omega_iov;
            diagnostics.omega_relative_spd_margin = omega_relative_spd_margin;
            diagnostics.omega_iov_relative_spd_margin = omega_iov_relative_spd_margin;
            diagnostics.residual_error_estimates = residual_error_estimates;
            diagnostics.conditional_negative_log_likelihood = conditional_negative_log_likelihood;
            diagnostics.eta_log_prior = eta_log_prior;
            diagnostics.kappa_log_prior = kappa_log_prior;
            diagnostics.covariate_betas = covariate_betas;
            diagnostics.covariate_beta_estimated = covariate_beta_estimated;
        }
        Ok(())
    }

    fn update_population_and_recenter_etas(&mut self) -> Result<Vec<f64>> {
        let old_population_phi = population_phi(
            &self.population_parameters,
            &self.initialization.parameter_scales,
        )?;
        let mut new_population_phi = old_population_phi.clone();
        for (parameter_index, parameter_phi) in new_population_phi.iter_mut().enumerate() {
            if self.initialization.estimated_parameters[parameter_index]
                && self
                    .initialization
                    .random_effect_indices
                    .contains(&parameter_index)
            {
                *parameter_phi = self.sufficient_statistics.mean_phi[parameter_index];
            }
        }

        for (eta_index, parameter_index) in self
            .initialization
            .random_effect_indices
            .iter()
            .copied()
            .enumerate()
        {
            let realized_shift =
                new_population_phi[parameter_index] - old_population_phi[parameter_index];
            for subject_chains in &mut self.etas {
                for eta in subject_chains {
                    eta[eta_index] -= realized_shift;
                }
            }
        }
        self.population_parameters =
            population_psi(&new_population_phi, &self.initialization.parameter_scales)?;
        Ok(new_population_phi)
    }

    fn update_covariate_population_and_recenter_etas(&mut self) -> Result<Array2<f64>> {
        let model = self
            .covariate_model
            .as_ref()
            .expect("covariate update requires a resolved model")
            .clone();
        let statistics = self
            .covariate_statistics
            .as_ref()
            .expect("covariate update requires sufficient statistics")
            .clone();
        let q = self.initialization.random_effect_indices.len();
        let old_population_phi = population_phi(
            &self.population_parameters,
            &self.initialization.parameter_scales,
        )?;
        let old_subject_mu = self
            .subject_mu_phi
            .as_ref()
            .expect("covariate update requires subject means")
            .clone();

        let free_intercepts = self
            .initialization
            .random_effect_indices
            .iter()
            .copied()
            .filter(|index| self.initialization.estimated_parameters[*index])
            .collect::<Vec<_>>();
        let free_effects = model
            .estimates()
            .iter()
            .enumerate()
            .filter_map(|(index, estimate)| {
                (estimate.estimated()
                    && self
                        .initialization
                        .random_effect_indices
                        .contains(&model.parameter_indices()[index]))
                .then_some(index)
            })
            .collect::<Vec<_>>();
        let width = free_intercepts.len() + free_effects.len();
        let random_row = self
            .initialization
            .random_effect_indices
            .iter()
            .enumerate()
            .map(|(row, parameter)| (*parameter, row))
            .collect::<BTreeMap<_, _>>();
        let mut designs = Vec::with_capacity(model.subject_design().len());
        let mut offsets = Vec::with_capacity(model.subject_design().len());
        for subject in model.subject_design() {
            let mut design = Array2::zeros((q, width));
            let mut offset = vec![0.0; q];
            for (row, parameter_index) in self
                .initialization
                .random_effect_indices
                .iter()
                .copied()
                .enumerate()
            {
                if let Some(column) = free_intercepts
                    .iter()
                    .position(|index| *index == parameter_index)
                {
                    design[[row, column]] = 1.0;
                } else {
                    offset[row] = old_population_phi[parameter_index];
                }
            }
            for (effect_index, value) in subject.values().iter().copied().enumerate() {
                let parameter_index = model.parameter_indices()[effect_index];
                let Some(&row) = random_row.get(&parameter_index) else {
                    continue;
                };
                if let Some(effect_column) =
                    free_effects.iter().position(|index| *index == effect_index)
                {
                    design[[row, free_intercepts.len() + effect_column]] = value;
                } else {
                    offset[row] += value * model.estimates()[effect_index].estimate();
                }
            }
            designs.push(design);
            offsets.push(offset);
        }

        let solution = if width == 0 {
            Vec::new()
        } else {
            solve_covariate_gls(CovariateGlsProblem {
                design: &designs,
                expected_phi: &statistics.expected_phi,
                offset: &offsets,
                omega: &self.omega,
            })?
        };
        let mut new_population_phi = old_population_phi;
        for (column, parameter_index) in free_intercepts.iter().copied().enumerate() {
            new_population_phi[parameter_index] = solution[column];
        }
        let mut beta_values = model
            .estimates()
            .iter()
            .map(|estimate| estimate.estimate())
            .collect::<Vec<_>>();
        for (column, effect_index) in free_effects.iter().copied().enumerate() {
            beta_values[effect_index] = solution[free_intercepts.len() + column];
        }
        let updated_model = model.with_estimates(&beta_values)?;
        let subject_population = updated_model.subject_population_parameters(
            &new_population_phi,
            &self.initialization.parameter_scales,
        )?;
        let new_subject_mu = subject_population
            .iter()
            .map(|row| row.phi().to_vec())
            .collect::<Vec<_>>();
        for (subject_index, subject_chains) in self.etas.iter_mut().enumerate() {
            let old_random = self
                .initialization
                .random_effect_indices
                .iter()
                .map(|index| old_subject_mu[subject_index][*index])
                .collect::<Vec<_>>();
            let new_random = self
                .initialization
                .random_effect_indices
                .iter()
                .map(|index| new_subject_mu[subject_index][*index])
                .collect::<Vec<_>>();
            for eta in subject_chains {
                rebase_eta(eta, &old_random, &new_random)?;
            }
        }
        let subject_mu_random = new_subject_mu
            .iter()
            .map(|mean| {
                self.initialization
                    .random_effect_indices
                    .iter()
                    .map(|index| mean[*index])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let candidate = if q == 0 {
            Array2::zeros((0, 0))
        } else {
            subject_centered_omega(
                &statistics.global_second_moment,
                &statistics.expected_phi,
                &subject_mu_random,
            )?
        };
        self.population_parameters =
            population_psi(&new_population_phi, &self.initialization.parameter_scales)?;
        self.subject_mu_phi = Some(new_subject_mu);
        self.covariate_model = Some(updated_model);
        Ok(candidate)
    }

    fn adapt_proposal_step_sizes(&mut self) {
        if self.steps_since_adapt < self.adapt_interval {
            return;
        }

        for parameter_index in 0..self.proposal_step_sizes.len() {
            let proposed = self.adaptation_proposal_counts[parameter_index].max(1);
            let acceptance_rate =
                self.adaptation_accept_counts[parameter_index] as f64 / proposed as f64;
            self.proposal_step_sizes[parameter_index] = adapt_component_step_size(
                self.proposal_step_sizes[parameter_index],
                acceptance_rate,
            );
            self.adaptation_accept_counts[parameter_index] = 0;
            self.adaptation_proposal_counts[parameter_index] = 0;
        }
        for subject_index in 0..self.eta_block_step_sizes.len() {
            let proposed = self.eta_block_adaptation_proposal_counts[subject_index].max(1);
            let acceptance_rate =
                self.eta_block_adaptation_accept_counts[subject_index] as f64 / proposed as f64;
            self.eta_block_step_sizes[subject_index] = adapt_block_step_size(
                self.eta_block_step_sizes[subject_index],
                acceptance_rate,
                ETA_BLOCK_TARGET_ACCEPTANCE,
            );
            self.eta_block_adaptation_accept_counts[subject_index] = 0;
            self.eta_block_adaptation_proposal_counts[subject_index] = 0;
        }
        for subject_index in 0..self.kappa_proposal_step_sizes.len() {
            let proposed = self.kappa_adaptation_proposal_counts[subject_index].max(1);
            let acceptance_rate =
                self.kappa_adaptation_accept_counts[subject_index] as f64 / proposed as f64;
            self.kappa_proposal_step_sizes[subject_index] = adapt_block_step_size(
                self.kappa_proposal_step_sizes[subject_index],
                acceptance_rate,
                KAPPA_BLOCK_TARGET_ACCEPTANCE,
            );
            self.kappa_adaptation_accept_counts[subject_index] = 0;
            self.kappa_adaptation_proposal_counts[subject_index] = 0;
        }
        self.steps_since_adapt = 0;
    }

    fn component_random_walk_eta(
        &mut self,
        current_eta: &[f64],
        parameter_index: usize,
    ) -> Vec<f64> {
        let mut proposed_eta = current_eta.to_vec();
        proposed_eta[parameter_index] +=
            self.proposal_step_sizes[parameter_index] * self.standard_normal();
        proposed_eta
    }

    fn prior_independence_eta(&mut self, lower: &[Vec<f64>]) -> Result<Vec<f64>> {
        let standard_normals = (0..lower.len())
            .map(|_| self.standard_normal())
            .collect::<Vec<_>>();
        correlated_random_walk(&vec![0.0; lower.len()], lower, &standard_normals, 1.0)
    }

    fn saemix_subset_groups(&mut self, n_parameters: usize) -> (usize, Vec<Vec<usize>>) {
        if n_parameters == 1 {
            return (1, vec![vec![0]]);
        }
        let subset_size = self.cycle % (n_parameters - 1) + 2;
        if subset_size == n_parameters {
            return (subset_size, vec![(0..n_parameters).collect()]);
        }

        let mut candidates = (1..n_parameters).collect::<Vec<_>>();
        for index in 0..(subset_size - 1) {
            let selected = self.rng.random_range(index..candidates.len());
            candidates.swap(index, selected);
        }
        let mut offsets = vec![0];
        offsets.extend_from_slice(&candidates[..subset_size - 1]);
        let groups = (0..n_parameters)
            .map(|start| {
                offsets
                    .iter()
                    .map(|offset| (start + offset) % n_parameters)
                    .collect()
            })
            .collect();
        (subset_size, groups)
    }

    fn subset_random_walk_eta(
        &mut self,
        current_eta: &[f64],
        parameters: &[usize],
        subset_size: usize,
    ) -> Vec<f64> {
        let mut proposed_eta = current_eta.to_vec();
        for parameter in parameters {
            let step = if current_eta.len() == 1 {
                self.proposal_step_sizes[*parameter]
            } else {
                self.saemix_subset_step_sizes[*parameter][subset_size - 1]
            };
            proposed_eta[*parameter] += step * self.standard_normal();
        }
        proposed_eta
    }

    fn saemix_map_distribution(
        &self,
        subject_index: usize,
        policy: SaemixMcmcConfig,
    ) -> Result<SaemixMapDistribution> {
        let n_eta = self.initialization.random_effect_indices.len();
        let initial = self.etas[subject_index][0].clone();
        let scales = (0..n_eta)
            .map(|index| self.omega[[index, index]].sqrt() * policy.map_initial_step)
            .collect::<Vec<_>>();
        let solution = optimize_conditional_mode(
            initial,
            &scales,
            policy.map_max_iterations as u64,
            policy.map_sd_tolerance,
            |eta| match self.score_subject_latents(subject_index, eta, &[]) {
                Ok(score) if score.log_posterior().is_finite() => -score.log_posterior(),
                _ => f64::INFINITY,
            },
        )?;
        let coordinates = (0..n_eta)
            .map(|index| JointLatentCoordinate {
                index,
                name: format!("eta:{}", self.initialization.random_effect_names[index]),
                kind: JointLatentCoordinateKind::Eta {
                    parameter_index: self.initialization.random_effect_indices[index],
                },
                prior_sd: self.omega[[index, index]].sqrt(),
            })
            .collect::<Vec<_>>();
        let prior_sds = coordinates
            .iter()
            .map(|coordinate| coordinate.prior_sd)
            .collect::<Vec<_>>();
        let mode_metadata = ConditionalModeMetadata {
            converged: solution.converged,
            iterations: solution.iterations,
            objective_value: solution.objective,
            termination_message: solution.termination,
        };
        let curvature = conditional_mode_curvature(
            &solution.coordinates,
            &prior_sds,
            &coordinates,
            &mode_metadata,
            |eta| match self.score_subject_latents(subject_index, eta, &[]) {
                Ok(score) if score.log_posterior().is_finite() => -score.log_posterior(),
                _ => f64::INFINITY,
            },
        );
        if !matches!(curvature.status, ConditionalCurvatureStatus::Available) {
            anyhow::bail!(
                "SAEMix q4 conditional curvature is unavailable for subject '{}': {:?}",
                self.initialization.subject_ids[subject_index],
                curvature.status
            );
        }
        let covariance_rows = curvature.latent_covariance.ok_or_else(|| {
            anyhow::anyhow!("available SAEMix q4 curvature lacks latent covariance")
        })?;
        let covariance = Array2::from_shape_vec(
            (n_eta, n_eta),
            covariance_rows.into_iter().flatten().collect(),
        )?;
        let lower = cholesky_lower(&covariance)?;
        Ok(SaemixMapDistribution {
            mode: solution.coordinates,
            covariance,
            lower,
        })
    }

    fn block_random_walk_eta(
        &mut self,
        current_eta: &[f64],
        subject_index: usize,
    ) -> Result<Vec<f64>> {
        let lower = cholesky_lower(&self.omega)?;
        let standard_normals = (0..current_eta.len())
            .map(|_| self.standard_normal())
            .collect::<Vec<_>>();
        correlated_random_walk(
            current_eta,
            &lower,
            &standard_normals,
            self.eta_block_step_sizes[subject_index],
        )
    }

    fn block_random_walk_kappa(
        &mut self,
        current_kappa: &[f64],
        subject_index: usize,
    ) -> Result<Vec<f64>> {
        let omega_iov = self
            .omega_iov
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("kappa proposal requires configured omega_iov"))?;
        let lower = cholesky_lower(omega_iov)?;
        let standard_normals = (0..current_kappa.len())
            .map(|_| self.standard_normal())
            .collect::<Vec<_>>();
        correlated_random_walk(
            current_kappa,
            &lower,
            &standard_normals,
            self.kappa_proposal_step_sizes[subject_index],
        )
    }

    fn standard_normal(&mut self) -> f64 {
        let u1 = self.rng.random::<f64>().max(f64::MIN_POSITIVE);
        let u2 = self.rng.random::<f64>();
        (-2.0_f64 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn accept_proposal(&mut self, log_acceptance_ratio: f64) -> bool {
        if !log_acceptance_ratio.is_finite() {
            return false;
        }
        if log_acceptance_ratio >= 0.0 {
            return true;
        }
        self.rng.random::<f64>().max(f64::MIN_POSITIVE).ln() < log_acceptance_ratio
    }

    fn individual_parameters(&self, subject_index: usize, chain_index: usize) -> Vec<f64> {
        self.individual_parameters_from_eta(subject_index, &self.etas[subject_index][chain_index])
            .expect("stored eta should match parameter dimensions")
    }

    fn individual_parameters_from_eta(
        &self,
        subject_index: usize,
        eta: &[f64],
    ) -> Result<Vec<f64>> {
        match self.subject_mu_phi.as_ref() {
            Some(means) => individual_psi_from_subject_mean(
                &means[subject_index],
                &self.initialization.parameter_scales,
                &self.initialization.random_effect_indices,
                eta,
            ),
            None => individual_psi(
                &self.population_parameters,
                &self.initialization.parameter_scales,
                &self.initialization.random_effect_indices,
                eta,
            ),
        }
    }

    fn individual_phi(&self, subject_index: usize, chain_index: usize) -> Result<Vec<f64>> {
        match self.subject_mu_phi.as_ref() {
            Some(means) => individual_phi_from_subject_mean(
                &means[subject_index],
                &self.initialization.random_effect_indices,
                &self.etas[subject_index][chain_index],
            ),
            None => individual_phi(
                &self.population_parameters,
                &self.initialization.parameter_scales,
                &self.initialization.random_effect_indices,
                &self.etas[subject_index][chain_index],
            ),
        }
    }

    fn current_phi_statistics(&self) -> Result<PhiSufficientStatistics> {
        let mut subject_phi = Vec::with_capacity(
            self.initialization.subject_ids.len() * self.initialization.n_chains,
        );
        for subject_index in 0..self.initialization.subject_ids.len() {
            for chain_index in 0..self.initialization.n_chains {
                subject_phi.push(self.individual_phi(subject_index, chain_index)?);
            }
        }
        PhiSufficientStatistics::from_subject_phi(&subject_phi)
    }

    fn current_covariate_statistics(&self) -> Result<CovariateSufficientStatistics> {
        let mut subjects = Vec::with_capacity(self.initialization.subject_ids.len());
        for subject_index in 0..self.initialization.subject_ids.len() {
            let mut chains = Vec::with_capacity(self.initialization.n_chains);
            for chain_index in 0..self.initialization.n_chains {
                let phi = self.individual_phi(subject_index, chain_index)?;
                chains.push(
                    self.initialization
                        .random_effect_indices
                        .iter()
                        .map(|index| phi[*index])
                        .collect(),
                );
            }
            subjects.push(chains);
        }
        CovariateSufficientStatistics::from_subject_chains(&subjects)
    }

    fn current_residual_statistics_and_information(
        &self,
    ) -> Result<(
        ResidualSufficientStatistics,
        std::result::Result<Vec<CompleteDerivative>, String>,
    )> {
        let mut total = ResidualSufficientStatistics::zero(self.error_models.len());
        let layout = self.information.layout();
        let mut replicates = (0..self.initialization.n_chains)
            .map(|_| CompleteDerivative::zero(layout.len()))
            .collect::<Vec<_>>();
        let mut information_error = None;
        // Preserve the established subject-major/chain-minor prediction and
        // accumulation order so this diagnostic cannot alter fit trajectories.
        for subject_index in 0..self.initialization.subject_ids.len() {
            let subject = self.data.subjects()[subject_index];
            for (chain_index, derivative) in replicates.iter_mut().enumerate() {
                if information_error.is_none() {
                    let derivative_result = match self.covariate_model.as_ref() {
                        Some(model) => derivative.add_covariate_population_prior(
                            &self.etas[subject_index][chain_index],
                            &self.omega,
                            &self.initialization.random_effect_indices,
                            model.parameter_indices(),
                            model.subject_design()[subject_index].values(),
                            layout,
                        ),
                        None => derivative.add_population_prior(
                            &self.etas[subject_index][chain_index],
                            &self.omega,
                            &self.initialization.random_effect_indices,
                            layout,
                        ),
                    };
                    if let Err(error) = derivative_result {
                        information_error = Some(error.to_string());
                    }
                }
                if self.omega_iov.is_none() {
                    let parameters = self.individual_parameters(subject_index, chain_index);
                    let predictions = self
                        .equation
                        .estimate_predictions_dense(subject, &parameters)?;
                    total.add_assign(&ResidualSufficientStatistics::from_predictions(
                        &predictions,
                        &self.error_models,
                    ));
                    if information_error.is_none() {
                        if let Err(error) =
                            derivative.add_predictions(&predictions, &self.error_models, layout)
                        {
                            information_error = Some(error.to_string());
                        }
                    }
                    continue;
                }
                for (occasion, kappa) in subject
                    .occasions()
                    .iter()
                    .zip(&self.kappas[subject_index][chain_index])
                {
                    if information_error.is_none() {
                        if let Some(omega_iov) = self.omega_iov.as_ref() {
                            if let Err(error) = derivative.add_iov_prior(kappa, omega_iov, layout) {
                                information_error = Some(error.to_string());
                            }
                        }
                    }
                    let parameters = match self.subject_mu_phi.as_ref() {
                        Some(means) => occasion_psi_from_subject_mean(
                            &means[subject_index],
                            &self.initialization.parameter_scales,
                            &self.initialization.random_effect_indices,
                            &self.etas[subject_index][chain_index],
                            &self.initialization.iov_effect_indices,
                            kappa,
                        ),
                        None => occasion_psi(
                            &self.population_parameters,
                            &self.initialization.parameter_scales,
                            &self.initialization.random_effect_indices,
                            &self.etas[subject_index][chain_index],
                            &self.initialization.iov_effect_indices,
                            kappa,
                        ),
                    }?;
                    let occasion_subject =
                        Subject::from_occasions(subject.id().to_owned(), vec![occasion.clone()]);
                    let predictions = self
                        .equation
                        .estimate_predictions_dense(&occasion_subject, &parameters)?;
                    total.add_assign(&ResidualSufficientStatistics::from_predictions(
                        &predictions,
                        &self.error_models,
                    ));
                    if information_error.is_none() {
                        if let Err(error) =
                            derivative.add_predictions(&predictions, &self.error_models, layout)
                        {
                            information_error = Some(error.to_string());
                        }
                    }
                }
            }
        }
        Ok((
            total,
            match information_error {
                Some(error) => Err(error),
                None => Ok(replicates),
            },
        ))
    }

    #[cfg(test)]
    fn current_residual_statistics(&self) -> Result<ResidualSufficientStatistics> {
        self.current_residual_statistics_and_information()
            .map(|(statistics, _)| statistics)
    }

    fn refresh_subject_scores_from_chains(&mut self) -> Result<()> {
        let n_chains = self.initialization.n_chains as f64;
        let mut subject_log_likelihoods = vec![0.0; self.initialization.subject_ids.len()];
        let mut subject_log_priors = vec![0.0; self.initialization.subject_ids.len()];
        let mut subject_kappa_log_priors = vec![0.0; self.initialization.subject_ids.len()];
        for subject_index in 0..self.initialization.subject_ids.len() {
            for chain_index in 0..self.initialization.n_chains {
                let score = self.score_subject_latents(
                    subject_index,
                    &self.etas[subject_index][chain_index],
                    &self.kappas[subject_index][chain_index],
                )?;
                subject_log_likelihoods[subject_index] += score.log_likelihood / n_chains;
                subject_log_priors[subject_index] += score.eta_log_prior / n_chains;
                subject_kappa_log_priors[subject_index] += score.kappa_log_prior / n_chains;
            }
        }
        self.subject_log_likelihoods = subject_log_likelihoods;
        self.subject_log_priors = subject_log_priors;
        self.subject_kappa_log_priors = subject_kappa_log_priors;
        Ok(())
    }

    fn score_subject_latents(
        &self,
        subject_index: usize,
        eta: &[f64],
        kappas: &[Vec<f64>],
    ) -> Result<SubjectPosteriorScore> {
        self.score_subject_latents_at(subject_index, eta, kappas, None)
    }

    fn non_iiv_coordinate_layout(&self) -> NonIivCoordinateLayout {
        let population_indices = self
            .initialization
            .estimated_parameters
            .iter()
            .enumerate()
            .filter_map(|(index, estimated)| {
                (*estimated && !self.initialization.random_effect_indices.contains(&index))
                    .then_some(index)
            })
            .collect();
        let covariate_indices = self
            .covariate_model
            .as_ref()
            .map(|model| {
                model
                    .estimates()
                    .iter()
                    .enumerate()
                    .filter_map(|(index, estimate)| {
                        (estimate.estimated()
                            && !self
                                .initialization
                                .random_effect_indices
                                .contains(&model.parameter_indices()[index]))
                        .then_some(index)
                    })
                    .collect()
            })
            .unwrap_or_default();
        NonIivCoordinateLayout {
            population_indices,
            covariate_indices,
        }
    }

    fn non_iiv_population_update_active(&self, parameter_step: f64) -> bool {
        let Some(post_burn_start) = self.initialization.schedule.pure_burn_in.checked_add(1) else {
            return false;
        };
        let first_active_cycle = self
            .initialization
            .schedule
            .variance_floor_iterations
            .max(post_burn_start);
        parameter_step.is_finite() && parameter_step > 0.0 && self.cycle >= first_active_cycle
    }

    fn pack_non_iiv_coordinates(&self, layout: &NonIivCoordinateLayout) -> Result<Vec<f64>> {
        let population = population_phi(
            &self.population_parameters,
            &self.initialization.parameter_scales,
        )?;
        let mut coordinates = layout
            .population_indices
            .iter()
            .map(|index| population[*index])
            .collect::<Vec<_>>();
        if let Some(model) = self.covariate_model.as_ref() {
            coordinates.extend(
                layout
                    .covariate_indices
                    .iter()
                    .map(|index| model.estimates()[*index].estimate()),
            );
        }
        Ok(coordinates)
    }

    fn non_iiv_candidate_components(
        &self,
        layout: &NonIivCoordinateLayout,
        coordinates: &[f64],
    ) -> Result<NonIivCandidateComponents> {
        if coordinates.len() != layout.len() || coordinates.iter().any(|value| !value.is_finite()) {
            anyhow::bail!("non-IIV population coordinate width or value is invalid");
        }
        let mut population = population_phi(
            &self.population_parameters,
            &self.initialization.parameter_scales,
        )?;
        for (coordinate, parameter_index) in coordinates
            .iter()
            .copied()
            .zip(layout.population_indices.iter().copied())
        {
            population[parameter_index] = coordinate;
        }
        let population_parameters =
            population_psi(&population, &self.initialization.parameter_scales)?;
        if !parameters_are_strictly_in_domain(
            &population_parameters,
            &self.initialization.parameter_scales,
        ) {
            anyhow::bail!("non-IIV population candidate violates its declared parameter domain");
        }

        let covariate_model = match self.covariate_model.as_ref() {
            Some(model) => {
                let mut values = model
                    .estimates()
                    .iter()
                    .map(|estimate| estimate.estimate())
                    .collect::<Vec<_>>();
                for (coordinate, effect_index) in coordinates[layout.population_indices.len()..]
                    .iter()
                    .copied()
                    .zip(layout.covariate_indices.iter().copied())
                {
                    values[effect_index] = coordinate;
                }
                Some(model.with_estimates(&values)?)
            }
            None if layout.covariate_indices.is_empty() => None,
            None => anyhow::bail!("non-IIV covariate coordinates lack a covariate model"),
        };
        let subject_rows = covariate_model
            .as_ref()
            .map(|model| {
                model.subject_population_parameters(
                    &population,
                    &self.initialization.parameter_scales,
                )
            })
            .transpose()?;
        if subject_rows.as_ref().is_some_and(|rows| {
            rows.iter().any(|row| {
                !parameters_are_strictly_in_domain(row.psi(), &self.initialization.parameter_scales)
            })
        }) {
            anyhow::bail!("non-IIV covariate candidate violates a declared parameter domain");
        }
        let subject_means = subject_rows.map(|rows| {
            rows.into_iter()
                .map(|row| row.phi().to_vec())
                .collect::<Vec<_>>()
        });
        Ok((population_parameters, covariate_model, subject_means))
    }

    pub(super) fn non_iiv_observation_nll(
        &self,
        layout: &NonIivCoordinateLayout,
        coordinates: &[f64],
    ) -> Result<f64> {
        let (population_parameters, _covariate_model, subject_means) =
            self.non_iiv_candidate_components(layout, coordinates)?;
        let chain_count = self.initialization.n_chains;
        if chain_count == 0 {
            anyhow::bail!("non-IIV observation objective requires at least one chain");
        }
        let mut objective = 0.0;
        for subject_index in 0..self.initialization.subject_ids.len() {
            let subject = self.data.subjects()[subject_index];
            let subject_mean = subject_means
                .as_ref()
                .map(|means| means[subject_index].as_slice());
            for chain_index in 0..chain_count {
                let eta = &self.etas[subject_index][chain_index];
                let log_likelihood = if self.omega_iov.is_none() {
                    let parameters = match subject_mean {
                        Some(mean) => individual_psi_from_subject_mean(
                            mean,
                            &self.initialization.parameter_scales,
                            &self.initialization.random_effect_indices,
                            eta,
                        ),
                        None => individual_psi(
                            &population_parameters,
                            &self.initialization.parameter_scales,
                            &self.initialization.random_effect_indices,
                            eta,
                        ),
                    }?;
                    parametric_subject_log_likelihood(
                        &self.equation,
                        subject,
                        &parameters,
                        &self.error_models,
                    )
                } else {
                    let kappas = &self.kappas[subject_index][chain_index];
                    if kappas.len() != subject.occasions().len() {
                        anyhow::bail!("non-IIV objective kappa/occasion dimension mismatch");
                    }
                    let mut value = 0.0;
                    for (occasion, kappa) in subject.occasions().iter().zip(kappas) {
                        let parameters = match subject_mean {
                            Some(mean) => occasion_psi_from_subject_mean(
                                mean,
                                &self.initialization.parameter_scales,
                                &self.initialization.random_effect_indices,
                                eta,
                                &self.initialization.iov_effect_indices,
                                kappa,
                            ),
                            None => occasion_psi(
                                &population_parameters,
                                &self.initialization.parameter_scales,
                                &self.initialization.random_effect_indices,
                                eta,
                                &self.initialization.iov_effect_indices,
                                kappa,
                            ),
                        }?;
                        let occasion_value = parametric_occasion_log_likelihood(
                            &self.equation,
                            subject.id(),
                            occasion,
                            &parameters,
                            &self.error_models,
                        );
                        if !occasion_value.is_finite() {
                            anyhow::bail!("non-IIV observation objective is non-finite");
                        }
                        value += occasion_value;
                    }
                    value
                };
                if !log_likelihood.is_finite() {
                    anyhow::bail!("non-IIV observation objective is non-finite");
                }
                objective -= log_likelihood / chain_count as f64;
            }
        }
        if !objective.is_finite() {
            anyhow::bail!("non-IIV observation objective is non-finite");
        }
        Ok(objective)
    }

    fn update_non_iiv_population(&mut self, parameter_step: f64) -> Result<bool> {
        let layout = self.non_iiv_coordinate_layout();
        if layout.is_empty() || !self.non_iiv_population_update_active(parameter_step) {
            return Ok(false);
        }

        let initial = self.pack_non_iiv_coordinates(&layout)?;
        let initial_objective = self.non_iiv_observation_nll(&layout, &initial)?;
        if !initial_objective.is_finite() {
            anyhow::bail!("current non-IIV observation objective is non-finite");
        }

        let mut simplex = Vec::with_capacity(initial.len() + 1);
        simplex.push(initial.clone());
        for coordinate in 0..initial.len() {
            let mut point = initial.clone();
            point[coordinate] += 0.1 * initial[coordinate].abs().max(1.0);
            simplex.push(point);
        }
        let solver = NelderMead::new(simplex).with_sd_tolerance(NON_IIV_OPTIMIZER_SD_TOLERANCE)?;
        let execution = Executor::new(
            NonIivPopulationCost {
                state: self,
                layout: &layout,
            },
            solver,
        )
        .configure(|state| state.max_iters(NON_IIV_OPTIMIZER_MAX_ITERATIONS))
        .run();
        let result = match execution {
            Ok(result) => result,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "Non-IIV population optimizer failed; retaining current state"
                );
                return Ok(false);
            }
        };
        let Some(candidate) = result.state.best_param.as_ref() else {
            return Ok(false);
        };
        let candidate_objective = match self.non_iiv_observation_nll(&layout, candidate) {
            Ok(value) if value.is_finite() => value,
            _ => return Ok(false),
        };
        if !non_iiv_candidate_improves(initial_objective, candidate_objective) {
            return Ok(false);
        }

        let applied = initial
            .iter()
            .zip(candidate)
            .map(|(current, target)| current + parameter_step * (target - current))
            .collect::<Vec<_>>();
        match self.non_iiv_observation_nll(&layout, &applied) {
            Ok(value) if value.is_finite() => {}
            _ => return Ok(false),
        }

        let (population_parameters, covariate_model, subject_means) =
            self.non_iiv_candidate_components(&layout, &applied)?;
        self.population_parameters = population_parameters;
        self.covariate_model = covariate_model;
        self.subject_mu_phi = subject_means;
        Ok(true)
    }

    fn score_subject_latents_at(
        &self,
        subject_index: usize,
        eta: &[f64],
        kappas: &[Vec<f64>],
        candidate: Option<&DiagnosticCandidate>,
    ) -> Result<SubjectPosteriorScore> {
        if eta.len() != self.initialization.random_effect_indices.len() {
            anyhow::bail!(
                "eta has {} values but there are {} random effects",
                eta.len(),
                self.initialization.random_effect_indices.len()
            );
        }

        let subject = self.data.subjects()[subject_index];
        let population_parameters = candidate
            .map_or(self.population_parameters.as_slice(), |value| {
                value.population_parameters.as_slice()
            });
        let omega = candidate.map_or(&self.omega, |value| &value.omega);
        let omega_iov = candidate.map_or(self.omega_iov.as_ref(), |value| value.omega_iov.as_ref());
        let error_models = candidate.map_or(&self.error_models, |value| &value.error_models);
        let candidate_covariates = candidate
            .and_then(|value| value.covariate_model.as_ref())
            .or(self.covariate_model.as_ref());
        let calculated_subject_mu = if candidate.is_some() {
            candidate_covariates
                .map(|model| {
                    let phi = population_phi(
                        population_parameters,
                        &self.initialization.parameter_scales,
                    )?;
                    Ok::<_, anyhow::Error>(
                        model.subject_population_parameters(
                            &phi,
                            &self.initialization.parameter_scales,
                        )?[subject_index]
                            .phi()
                            .to_vec(),
                    )
                })
                .transpose()?
        } else {
            None
        };
        let subject_mu = calculated_subject_mu.as_deref().or_else(|| {
            self.subject_mu_phi
                .as_ref()
                .map(|means| means[subject_index].as_slice())
        });
        let eta_log_prior = eta_log_prior_from_omega(eta, omega)?;
        if omega_iov.is_none() {
            let parameters = match subject_mu {
                Some(mean) => individual_psi_from_subject_mean(
                    mean,
                    &self.initialization.parameter_scales,
                    &self.initialization.random_effect_indices,
                    eta,
                ),
                None => individual_psi(
                    population_parameters,
                    &self.initialization.parameter_scales,
                    &self.initialization.random_effect_indices,
                    eta,
                ),
            }?;
            return Ok(SubjectPosteriorScore {
                log_likelihood: parametric_subject_log_likelihood(
                    &self.equation,
                    subject,
                    &parameters,
                    error_models,
                ),
                eta_log_prior,
                kappa_log_prior: 0.0,
            });
        }

        if kappas.len() != subject.occasions().len() {
            anyhow::bail!(
                "subject '{}' has {} occasions but {} kappa states",
                subject.id(),
                subject.occasions().len(),
                kappas.len()
            );
        }
        let omega_iov = omega_iov.expect("checked above");
        let mut log_likelihood = 0.0;
        let mut kappa_log_prior = 0.0;
        for (occasion, kappa) in subject.occasions().iter().zip(kappas) {
            let parameters = match subject_mu {
                Some(mean) => occasion_psi_from_subject_mean(
                    mean,
                    &self.initialization.parameter_scales,
                    &self.initialization.random_effect_indices,
                    eta,
                    &self.initialization.iov_effect_indices,
                    kappa,
                ),
                None => occasion_psi(
                    population_parameters,
                    &self.initialization.parameter_scales,
                    &self.initialization.random_effect_indices,
                    eta,
                    &self.initialization.iov_effect_indices,
                    kappa,
                ),
            }?;
            let occasion_log_likelihood = parametric_occasion_log_likelihood(
                &self.equation,
                subject.id(),
                occasion,
                &parameters,
                error_models,
            );
            if !occasion_log_likelihood.is_finite() {
                log_likelihood = f64::NEG_INFINITY;
            } else if log_likelihood.is_finite() {
                log_likelihood += occasion_log_likelihood;
            }
            kappa_log_prior += eta_log_prior_from_omega(kappa, omega_iov)?;
        }

        Ok(SubjectPosteriorScore {
            log_likelihood,
            eta_log_prior,
            kappa_log_prior,
        })
    }

    fn proposal_log_acceptance_ratio(
        &self,
        subject_index: usize,
        chain_index: usize,
        proposed_eta: &[f64],
    ) -> Result<f64> {
        let current = self.score_subject_latents(
            subject_index,
            &self.etas[subject_index][chain_index],
            &self.kappas[subject_index][chain_index],
        )?;
        let proposed = self.score_subject_latents(
            subject_index,
            proposed_eta,
            &self.kappas[subject_index][chain_index],
        )?;
        Ok(current.log_acceptance_ratio(proposed))
    }

    fn kappa_proposal_log_acceptance_ratio(
        &self,
        subject_index: usize,
        chain_index: usize,
        occasion_index: usize,
        proposed_kappa: &[f64],
    ) -> Result<f64> {
        let current_kappas = &self.kappas[subject_index][chain_index];
        let current = self.score_subject_latents(
            subject_index,
            &self.etas[subject_index][chain_index],
            current_kappas,
        )?;
        let mut proposed_kappas = current_kappas.clone();
        proposed_kappas[occasion_index] = proposed_kappa.to_vec();
        let proposed = self.score_subject_latents(
            subject_index,
            &self.etas[subject_index][chain_index],
            &proposed_kappas,
        )?;
        Ok(current.log_acceptance_ratio(proposed))
    }
}

mod diagnostics;
mod runner;
mod support;

use support::*;

#[cfg(test)]
mod tests;
