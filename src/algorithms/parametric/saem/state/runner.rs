use super::*;
use crate::algorithms::parametric::{NumericalFailurePhase, ParametricRunner};
use crate::estimation::parametric::information::derive_population_uncertainty;
use crate::estimation::parametric::marginal_likelihood::MarginalLikelihoodStatus;
use crate::estimation::parametric::shrinkage::{
    derive_eta_map_shrinkage, derive_eta_posterior_mean_shrinkage, derive_kappa_map_shrinkage,
    derive_kappa_posterior_mean_shrinkage, ShrinkageDiagnostics,
};
use crate::results::{derive_information_criteria, ParametricResult, SubjectEtaEstimate};

impl<E: Equation + Send + 'static> ParametricRunner<E> for SaemState<E> {
    fn step(&mut self) -> Result<Status> {
        if self.status.is_stop() {
            return Ok(self.status.clone());
        }

        if self.cycle >= self.initialization.schedule.total_iterations {
            self.status = Status::Stop(StopReason::MaxCycles);
            return Ok(self.status.clone());
        }

        self.cycle += 1;
        if let Err(error) = self.e_step() {
            let failure = NumericalFailure::new(
                self.cycle,
                NumericalFailurePhase::Expectation,
                format!("{error:#}"),
            );
            self.status = Status::Stop(StopReason::NumericalFailure);
            self.numerical_failure = Some(failure.clone());
            return Err(failure.into());
        }
        // m_step also accumulates damped covariance sufficient statistics
        // during pure burn-in while leaving theta, Omega, Omega_IOV, and sigma
        // unchanged, so it must run in every schedule phase.
        if let Err(error) = self.m_step() {
            let failure = NumericalFailure::new(
                self.cycle,
                NumericalFailurePhase::Maximization,
                format!("{error:#}"),
            );
            self.status = Status::Stop(StopReason::NumericalFailure);
            self.numerical_failure = Some(failure.clone());
            return Err(failure.into());
        }

        if self.cycle >= self.initialization.schedule.total_iterations {
            self.status = Status::Stop(StopReason::MaxCycles);
            let scheduled = self
                .operational_settings
                .zip(self.iterate_average.as_ref())
                .is_some_and(|(policy, average)| {
                    average.count >= policy.first_eligible_averaged_iteration
                        && (average.count - policy.first_eligible_averaged_iteration)
                            .is_multiple_of(policy.check_interval)
                });
            self.evaluate_operational_convergence(self.cycle, scheduled, true)?;
        } else {
            self.evaluate_operational_convergence(self.cycle, true, false)?;
        }

        Ok(self.status.clone())
    }

    fn request_stop(&mut self, reason: StopReason) {
        if self.status.is_continue() && self.numerical_failure.is_none() {
            self.status = Status::Stop(reason);
        }
    }

    fn cycle(&self) -> usize {
        self.cycle
    }

    fn status(&self) -> &Status {
        &self.status
    }

    fn cycle_diagnostics(&self) -> &[SaemCycleDiagnostics] {
        &self.cycle_diagnostics
    }

    fn log_likelihood(&self) -> f64 {
        self.subject_log_likelihoods.iter().sum()
    }

    fn population_parameters(&self) -> &[f64] {
        &self.population_parameters
    }

    fn covariate_betas(&self) -> Option<Vec<f64>> {
        self.covariate_model.as_ref().map(|model| {
            model
                .estimates()
                .iter()
                .map(|estimate| estimate.estimate())
                .collect()
        })
    }

    fn random_effect_names(&self) -> &[String] {
        &self.initialization.random_effect_names
    }

    fn iov_effect_names(&self) -> Option<&[String]> {
        (!self.initialization.iov_effect_names.is_empty())
            .then_some(&self.initialization.iov_effect_names)
    }

    fn eta_log_prior(&self) -> f64 {
        self.subject_log_priors.iter().sum()
    }

    fn kappa_log_prior(&self) -> f64 {
        self.subject_kappa_log_priors.iter().sum()
    }

    fn acceptance_rate(&self) -> Option<f64> {
        self.last_acceptance_rate
    }

    fn eta_block_acceptance_rate(&self) -> Option<f64> {
        self.last_eta_block_acceptance_rate
    }

    fn kappa_acceptance_rate(&self) -> Option<f64> {
        self.last_kappa_acceptance_rate
    }

    fn rejected_proposals(&self) -> Option<usize> {
        self.last_rejected_proposals
    }

    fn non_finite_proposals(&self) -> Option<usize> {
        self.last_non_finite_proposals
    }

    fn parameter_acceptance_rates(&self) -> Option<&[f64]> {
        self.last_acceptance_rate
            .map(|_| self.last_parameter_acceptance_rates.as_slice())
    }

    fn proposal_step_sizes(&self) -> Option<&[f64]> {
        Some(&self.proposal_step_sizes)
    }

    fn eta_block_step_sizes(&self) -> Option<&[f64]> {
        (self.eta_block_iterations > 0).then_some(self.eta_block_step_sizes.as_slice())
    }

    fn log_acceptance_ratios(&self) -> Option<&[f64]> {
        Some(&self.last_log_acceptance_ratios)
    }

    fn negative_log_likelihood(&self) -> f64 {
        self.negative_log_likelihood
    }

    fn n_chains(&self) -> Option<usize> {
        self.etas
            .first()
            .map(|subject_chains| subject_chains.len())
            .or(Some(self.initialization.n_chains))
    }

    fn omega(&self) -> Option<&Array2<f64>> {
        Some(&self.omega)
    }

    fn omega_iov(&self) -> Option<&Array2<f64>> {
        self.omega_iov.as_ref()
    }

    fn residual_sigmas(&self) -> &[f64] {
        &self.residual_sigmas
    }

    fn step_size(&self) -> f64 {
        self.initialization
            .schedule
            .stochastic_approximation_step(self.cycle)
    }

    fn total_iterations(&self) -> usize {
        self.initialization.schedule.total_iterations
    }

    fn into_result(mut self: Box<Self>) -> Result<ParametricResult<E>> {
        if let Some(failure) = self.numerical_failure.as_ref() {
            return Err(failure.clone().into());
        }

        let result_cycle = self.cycle;
        let estimator_metadata = match self.config.estimator_policy {
            SaemEstimatorPolicy::TerminalIterate => SaemEstimatorMetadata::default(),
            SaemEstimatorPolicy::AveragedIterates { .. } => {
                self.install_iterate_average().map_err(|error| {
                    NumericalFailure::new(
                        result_cycle,
                        NumericalFailurePhase::ResultAssembly,
                        format!("{error:#}"),
                    )
                })?
            }
        };
        let information_diagnostics = self.information.diagnostics();
        let population_uncertainty = derive_population_uncertainty(&information_diagnostics);
        let markov_simulation_variance = if self.operational_settings.is_some() {
            self.operational_diagnostics
                .checks
                .last()
                .and_then(|check| check.markov.clone())
                .unwrap_or_else(MarkovSimulationVarianceDiagnostics::disabled)
        } else {
            self.markov_variance_diagnostics(&estimator_metadata, &information_diagnostics)
        };
        let (conditional_modes, conditional_mode_error) = match conditional_modes(&self) {
            Ok(modes) => (modes, None),
            Err(error) if self.config.marginal_likelihood.is_some() => {
                (Vec::new(), Some(format!("{error:#}")))
            }
            Err(error) => {
                return Err(NumericalFailure::new(
                    result_cycle,
                    NumericalFailurePhase::ResultAssembly,
                    format!("{error:#}"),
                )
                .into())
            }
        };
        let marginal_likelihood = calculate_result_marginal_likelihood(
            &self,
            &conditional_modes,
            conditional_mode_error.as_deref(),
        );
        let information_criteria = derive_information_criteria(
            marginal_likelihood.as_ref(),
            &information_diagnostics.coordinates,
            self.initialization.subject_ids.len(),
        );
        let eta_chain_means = self
            .initialization
            .subject_ids
            .iter()
            .enumerate()
            .map(|(subject_index, subject_id)| {
                Ok(SubjectEtaEstimate {
                    subject_id: subject_id.clone(),
                    values: mean_vectors(
                        self.etas[subject_index].iter().map(|eta| eta.as_slice()),
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()
            .map_err(|error| {
                NumericalFailure::new(
                    result_cycle,
                    NumericalFailurePhase::ResultAssembly,
                    format!("{error:#}"),
                )
            })?;
        let mut kappa_chain_means = Vec::new();
        if self.omega_iov.is_some() {
            for (subject_index, subject_id) in self.initialization.subject_ids.iter().enumerate() {
                for (occasion_position, occasion) in self.data.subjects()[subject_index]
                    .occasions()
                    .iter()
                    .enumerate()
                {
                    kappa_chain_means.push(OccasionKappaEstimate {
                        subject_id: subject_id.clone(),
                        occasion_index: occasion.index(),
                        values: mean_vectors(
                            self.kappas[subject_index]
                                .iter()
                                .map(|chain| chain[occasion_position].as_slice()),
                        )
                        .map_err(|error| {
                            NumericalFailure::new(
                                result_cycle,
                                NumericalFailurePhase::ResultAssembly,
                                format!("{error:#}"),
                            )
                        })?,
                    });
                }
            }
        }
        let eta_variances = (0..self.omega.nrows())
            .map(|index| self.omega[[index, index]])
            .collect::<Vec<_>>();
        let eta_posterior_rows = eta_chain_means
            .iter()
            .map(|estimate| estimate.values.clone())
            .collect::<Vec<_>>();
        let eta_map_rows = (!conditional_modes.is_empty()).then(|| {
            conditional_modes
                .iter()
                .map(|mode| mode.eta.clone())
                .collect::<Vec<_>>()
        });
        let kappa_variances = self
            .omega_iov
            .as_ref()
            .map(|omega| {
                (0..omega.nrows())
                    .map(|index| omega[[index, index]])
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let kappa_posterior_rows = kappa_chain_means
            .iter()
            .map(|estimate| estimate.values.clone())
            .collect::<Vec<_>>();
        let kappa_map_rows = (!conditional_modes.is_empty()).then(|| {
            conditional_modes
                .iter()
                .flat_map(|mode| mode.kappas.iter().map(|kappa| kappa.values.clone()))
                .collect::<Vec<_>>()
        });
        let shrinkage = ShrinkageDiagnostics {
            eta_posterior_mean: derive_eta_posterior_mean_shrinkage(
                &self.initialization.random_effect_names,
                &eta_variances,
                &eta_posterior_rows,
            ),
            eta_map: derive_eta_map_shrinkage(
                &self.initialization.random_effect_names,
                &eta_variances,
                eta_map_rows.as_deref(),
            ),
            kappa_posterior_mean: derive_kappa_posterior_mean_shrinkage(
                &self.initialization.iov_effect_names,
                &kappa_variances,
                &kappa_posterior_rows,
            ),
            kappa_map: derive_kappa_map_shrinkage(
                &self.initialization.iov_effect_names,
                &kappa_variances,
                kappa_map_rows.as_deref(),
            ),
        };
        let residual_error_estimates = self.residual_error_estimates();
        let mut warnings =
            parametric_warnings(&self.cycle_diagnostics, self.config.covariance_stability);
        if let Some(diagnostics) = marginal_likelihood.as_ref() {
            match &diagnostics.status {
                MarginalLikelihoodStatus::Unavailable { failures } => {
                    warnings.push(ParametricWarning::MarginalLikelihoodUnavailable {
                        subjects: failures
                            .iter()
                            .map(|failure| failure.subject_id.clone())
                            .collect(),
                    });
                }
                MarginalLikelihoodStatus::AvailableWithNonconvergedModes { subjects } => {
                    warnings.push(ParametricWarning::MarginalLikelihoodNonconvergedModes {
                        subjects: subjects.clone(),
                    });
                }
                MarginalLikelihoodStatus::Available => {}
            }
        }
        let omega_structural_mask = self.initialization.omega.structural_mask().clone();
        let omega_estimated_mask = self.initialization.omega.estimated_mask().clone();
        let omega_iov_structural_mask = self
            .initialization
            .omega_iov
            .as_ref()
            .map(|omega| omega.structural_mask().clone());
        let omega_iov_estimated_mask = self
            .initialization
            .omega_iov
            .as_ref()
            .map(|omega| omega.estimated_mask().clone());
        let individual_estimates = if conditional_modes.is_empty() {
            self.initialization
                .subject_ids
                .iter()
                .enumerate()
                .map(|(subject_index, subject_id)| {
                    (
                        subject_id.clone(),
                        self.individual_parameters(subject_index, 0),
                    )
                })
                .collect()
        } else {
            conditional_modes
                .iter()
                .map(|mode| (mode.subject_id.clone(), mode.parameters.clone()))
                .collect()
        };

        let SaemState {
            equation,
            data,
            config,
            negative_log_likelihood: final_negative_log_likelihood,
            initialization,
            cycle,
            status,
            population_parameters,
            omega,
            omega_iov,
            residual_sigmas,
            cycle_diagnostics,
            operational_diagnostics,
            covariate_model,
            ..
        } = *self;

        Ok(ParametricResult {
            equation,
            data,
            config,
            effective_n_chains: initialization.n_chains,
            objective_function: 2.0 * final_negative_log_likelihood,
            converged: status.converged(),
            termination_reason: status.stop_reason().cloned(),
            iterations: cycle,
            subject_count: initialization.subject_ids.len(),
            observation_count: initialization.observation_count,
            parameter_names: initialization.parameter_names,
            parameter_scales: initialization.parameter_scales,
            estimated_parameters: initialization.estimated_parameters,
            population_initial: initialization.initial_population_parameters.clone(),
            population_estimates: population_parameters,
            random_effect_indices: initialization.random_effect_indices,
            random_effect_names: initialization.random_effect_names,
            omega,
            omega_structural_mask,
            omega_estimated_mask,
            omega_initial: initialization.omega.initial().clone(),
            iov_effect_indices: initialization.iov_effect_indices,
            iov_effect_names: initialization.iov_effect_names,
            omega_iov,
            omega_iov_structural_mask,
            omega_iov_estimated_mask,
            omega_iov_initial: initialization
                .omega_iov
                .as_ref()
                .map(|omega| omega.initial().clone()),
            residual_sigmas,
            residual_error_estimates,
            residual_initial_values: initialization.initial_residual_values.clone(),
            residual_initial_estimated: initialization.initial_residual_estimated.clone(),
            eta_chain_means,
            kappa_chain_means,
            conditional_modes,
            shrinkage,
            cycle_diagnostics,
            warnings,
            information_diagnostics,
            population_uncertainty,
            markov_simulation_variance,
            operational_diagnostics,
            marginal_likelihood,
            information_criteria,
            estimator_metadata,
            individual_estimates,
            covariate_model,
        })
    }
}
