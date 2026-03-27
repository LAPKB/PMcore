//! FOCEI (First-Order Conditional Estimation with Interaction) algorithm.

use std::collections::HashMap;

use anyhow::Result;
use faer::linalg::solvers::DenseSolveCore;
use faer::{Col, Mat};
use ndarray::Array2;
use pharmsol::{Data, Equation, ResidualErrorModels};

use crate::algorithms::{Status, StopReason};
use crate::estimation::parametric::{
    assemble_focei_result, batch_log_likelihood_from_eta, covariance_from_individual_etas,
    covariance_from_subject_means, covariate_state, ensure_positive_definite_covariance,
    estimate_beta, occasion_covariate_maps, recenter_individual_estimates, subject_covariate_maps,
    subject_mean_phi, subject_objective_from_eta, FoceiResultInput, Individual,
    IndividualEstimates, LikelihoodEstimates, ParameterTransform, ParametricIterationLog,
    ParametricWorkspace, Population,
};
use crate::model::{CovariateModel, CovariateSpec};
use crate::output::shared::RunConfiguration;

use super::algorithm::{ParametricAlgorithm, ParametricAlgorithmInput, ParametricConfig};

pub struct FoceiAlgorithm<E: Equation> {
    equation: E,
    data: Data,
    run_configuration: RunConfiguration,
    population: Population,
    individual_estimates: IndividualEstimates,
    iteration: usize,
    objf: f64,
    prev_objf: f64,
    status: Status,
    config: ParametricConfig,
    transforms: Vec<ParameterTransform>,
    residual_error_models: ResidualErrorModels,
    iteration_log: ParametricIterationLog,
    subject_covariate_model: Option<CovariateModel>,
    subject_covariates: Vec<HashMap<String, f64>>,
    occasion_covariate_model: Option<CovariateModel>,
    occasion_covariates: Vec<HashMap<String, f64>>,
}

impl<E: Equation + Send + 'static> FoceiAlgorithm<E> {
    pub(crate) fn create(input: ParametricAlgorithmInput<E>) -> Result<Box<Self>> {
        let run_configuration = input.run_configuration();
        let population = input.initial_population()?;
        let transforms = input.parameter_transforms();
        let residual_error_models = input.residual_error_models.clone();
        let config = ParametricConfig {
            max_iterations: input.runtime.cycles.max(1),
            objective_tolerance: input.runtime.convergence.likelihood,
            ..ParametricConfig::default()
        };
        let ParametricAlgorithmInput {
            equation,
            data,
            covariates,
            structured_covariates,
            ..
        } = input;
        let (
            mut subject_covariate_model,
            occasion_covariate_model,
            subject_covariates,
            occasion_covariates,
        ) = match covariates {
            CovariateSpec::InEquation => (None, None, Vec::new(), Vec::new()),
            CovariateSpec::Structured(spec) => (
                spec.subject_effects,
                spec.occasion_effects,
                subject_covariate_maps(&structured_covariates),
                occasion_covariate_maps(&structured_covariates),
            ),
        };

        if let Some(model) = subject_covariate_model.as_mut() {
            let initialize_intercepts =
                (0..model.beta().nrows()).all(|index| model.beta()[index].abs() < 1e-12);
            if initialize_intercepts {
                let intercepts = (0..population.npar())
                    .map(|index| population.mu()[index])
                    .collect::<Vec<_>>();
                model.set_intercepts(&intercepts)?;
            }
        }

        Ok(Box::new(Self {
            equation,
            data,
            run_configuration,
            population,
            individual_estimates: IndividualEstimates::new(),
            iteration: 0,
            objf: f64::INFINITY,
            prev_objf: f64::INFINITY,
            status: Status::Continue,
            config,
            transforms,
            residual_error_models,
            iteration_log: ParametricIterationLog::new(),
            subject_covariate_model,
            subject_covariates,
            occasion_covariate_model,
            occasion_covariates,
        }))
    }

    fn current_eta_matrix(&self) -> Array2<f64> {
        let n_subjects = self.data.subjects().len();
        let n_params = self.population.npar();
        let mut eta_matrix = Array2::zeros((n_subjects, n_params));

        for subject_index in 0..n_subjects {
            if let Some(individual) = self.individual_estimates.get(subject_index) {
                for param_index in 0..n_params {
                    eta_matrix[[subject_index, param_index]] = individual.eta()[param_index];
                }
            }
        }

        eta_matrix
    }

    fn invert_omega(&self) -> Mat<f64> {
        match self.population.omega().clone().llt(faer::Side::Lower) {
            Ok(cholesky) => cholesky.inverse(),
            Err(_) => {
                let n_params = self.population.npar();
                Mat::from_fn(n_params, n_params, |row, col| {
                    if row == col {
                        1.0 / self.population.omega()[(row, row)].max(1e-8)
                    } else {
                        0.0
                    }
                })
            }
        }
    }

    fn find_map_estimate(
        &self,
        subject_index: usize,
        eta_matrix: &mut Array2<f64>,
        omega_inv: &Mat<f64>,
        subject_means: &[Col<f64>],
    ) -> Result<Individual> {
        let n_params = self.population.npar();
        let mut current_ll = batch_log_likelihood_from_eta(
            &self.equation,
            &self.data,
            &self.residual_error_models,
            &self.transforms,
            eta_matrix,
            subject_means,
        )?[subject_index];
        let mut current_objf =
            subject_objective_from_eta(subject_index, eta_matrix, current_ll, omega_inv);
        let mut step_sizes = (0..n_params)
            .map(|param_index| {
                self.population.omega()[(param_index, param_index)]
                    .sqrt()
                    .max(1e-3)
                    * 0.5
            })
            .collect::<Vec<_>>();

        for _ in 0..4 {
            let mut improved = false;

            for param_index in 0..n_params {
                let baseline = eta_matrix[[subject_index, param_index]];
                let mut best_value = baseline;
                let mut best_ll = current_ll;
                let mut best_objf = current_objf;

                for direction in [-1.0, 1.0] {
                    let mut candidate = eta_matrix.clone();
                    candidate[[subject_index, param_index]] =
                        baseline + direction * step_sizes[param_index];
                    let proposed_ll = batch_log_likelihood_from_eta(
                        &self.equation,
                        &self.data,
                        &self.residual_error_models,
                        &self.transforms,
                        &candidate,
                        subject_means,
                    )?[subject_index];
                    let proposed_objf = subject_objective_from_eta(
                        subject_index,
                        &candidate,
                        proposed_ll,
                        omega_inv,
                    );

                    if proposed_objf < best_objf {
                        best_value = candidate[[subject_index, param_index]];
                        best_ll = proposed_ll;
                        best_objf = proposed_objf;
                    }
                }

                if best_objf < current_objf {
                    eta_matrix[[subject_index, param_index]] = best_value;
                    current_ll = best_ll;
                    current_objf = best_objf;
                    improved = true;
                }
            }

            if improved {
                continue;
            }

            for step_size in &mut step_sizes {
                *step_size *= 0.5;
            }

            if step_sizes.iter().all(|step_size| *step_size < 1e-6) {
                break;
            }
        }

        let eta = Col::from_fn(n_params, |param_index| {
            eta_matrix[[subject_index, param_index]]
        });
        let phi = Col::from_fn(n_params, |param_index| {
            subject_means[subject_index][param_index] + eta[param_index]
        });
        let mut individual =
            Individual::new(self.data.subjects()[subject_index].id().clone(), eta, phi)?;
        individual.set_objective_function(current_objf);
        Ok(individual)
    }

    fn update_population_parameters(&mut self) -> Result<()> {
        let n_subjects = self.individual_estimates.nsubjects();
        let n_params = self.population.npar();

        if n_subjects == 0 {
            return Ok(());
        }

        if let Some(model) = self.subject_covariate_model.clone() {
            let target_beta =
                estimate_beta(&model, &self.subject_covariates, &self.individual_estimates)?;
            let mut updated_model = model;
            updated_model.set_beta(target_beta)?;

            let mu = Col::from_fn(n_params, |index| {
                updated_model
                    .intercept(index)
                    .unwrap_or(self.population.mu()[index])
            });
            let subject_means = subject_mean_phi(
                &mu,
                n_subjects,
                Some(&updated_model),
                &self.subject_covariates,
            );

            self.population.update_mu(mu)?;
            self.individual_estimates =
                recenter_individual_estimates(&self.individual_estimates, &subject_means)?;
            self.subject_covariate_model = Some(updated_model);

            let omega = covariance_from_subject_means(&self.individual_estimates, &subject_means)?;
            self.population
                .update_omega(ensure_positive_definite_covariance(&omega))?;
        } else {
            let mu = Col::from_fn(n_params, |param_index| {
                self.individual_estimates
                    .iter()
                    .map(|individual| individual.psi()[param_index])
                    .sum::<f64>()
                    / n_subjects as f64
            });

            let subject_means = subject_mean_phi(&mu, n_subjects, None, &self.subject_covariates);
            self.population.update_mu(mu)?;
            self.individual_estimates =
                recenter_individual_estimates(&self.individual_estimates, &subject_means)?;

            let omega = covariance_from_individual_etas(&self.individual_estimates);
            self.population
                .update_omega(ensure_positive_definite_covariance(&omega))?;
        }

        let eta_matrix = self.current_eta_matrix();
        let subject_means = subject_mean_phi(
            self.population.mu(),
            self.data.subjects().len(),
            self.subject_covariate_model.as_ref(),
            &self.subject_covariates,
        );
        let omega_inv = self.invert_omega();
        let log_likelihoods = batch_log_likelihood_from_eta(
            &self.equation,
            &self.data,
            &self.residual_error_models,
            &self.transforms,
            &eta_matrix,
            &subject_means,
        )?;
        let mut updated = Vec::with_capacity(n_subjects);
        let mut total_objf = 0.0;

        for subject_index in 0..n_subjects {
            let individual = self.individual_estimates.get(subject_index).unwrap();
            let objf = subject_objective_from_eta(
                subject_index,
                &eta_matrix,
                log_likelihoods[subject_index],
                &omega_inv,
            );
            let mut rebuilt = Individual::new(
                individual.subject_id().to_string(),
                individual.eta().clone(),
                individual.psi().clone(),
            )?;
            rebuilt.set_objective_function(objf);
            total_objf += objf;
            updated.push(rebuilt);
        }

        self.individual_estimates = IndividualEstimates::from_vec(updated);
        self.prev_objf = self.objf;
        self.objf = total_objf;

        Ok(())
    }
}

impl<E: Equation + Send + 'static> ParametricAlgorithm<E> for FoceiAlgorithm<E> {
    fn equation(&self) -> &E {
        &self.equation
    }

    fn data(&self) -> &Data {
        &self.data
    }

    fn population(&self) -> &Population {
        &self.population
    }

    fn population_mut(&mut self) -> &mut Population {
        &mut self.population
    }

    fn individual_estimates(&self) -> &IndividualEstimates {
        &self.individual_estimates
    }

    fn iteration(&self) -> usize {
        self.iteration
    }

    fn increment_iteration(&mut self) -> usize {
        self.iteration += 1;
        self.iteration
    }

    fn objective_function(&self) -> f64 {
        self.objf
    }

    fn status(&self) -> &Status {
        &self.status
    }

    fn set_status(&mut self, status: Status) {
        self.status = status;
    }

    fn e_step(&mut self) -> Result<()> {
        let subjects = self.data.subjects();
        let mut individuals = Vec::with_capacity(subjects.len());
        let mut eta_matrix = self.current_eta_matrix();
        let omega_inv = self.invert_omega();
        let subject_means = subject_mean_phi(
            self.population.mu(),
            self.data.subjects().len(),
            self.subject_covariate_model.as_ref(),
            &self.subject_covariates,
        );

        for (subject_index, _subject) in subjects.iter().enumerate() {
            let individual =
                self.find_map_estimate(subject_index, &mut eta_matrix, &omega_inv, &subject_means)?;
            individuals.push(individual);
        }

        self.individual_estimates = IndividualEstimates::from_vec(individuals);
        Ok(())
    }

    fn m_step(&mut self) -> Result<()> {
        self.update_population_parameters()
    }

    fn evaluate(&mut self) -> Result<Status> {
        if std::path::Path::new("stop").exists() {
            self.status = Status::Stop(StopReason::Stopped);
            return Ok(self.status.clone());
        }

        if self.iteration >= self.config.max_iterations {
            self.status = Status::Stop(StopReason::MaxCycles);
            return Ok(self.status.clone());
        }

        if self.prev_objf.is_finite() {
            let objf_change = (self.objf - self.prev_objf).abs();
            let relative_change = objf_change / self.prev_objf.abs().max(1.0);
            if relative_change < self.config.objective_tolerance {
                self.status = Status::Stop(StopReason::Converged);
                return Ok(self.status.clone());
            }
        }

        if !self.objf.is_finite() {
            self.status = Status::Stop(StopReason::Converged);
            return Ok(self.status.clone());
        }

        self.status = Status::Continue;
        Ok(self.status.clone())
    }

    fn log_iteration(&mut self) {
        self.iteration_log
            .log_iteration(self.iteration, self.objf, &self.population, &self.status);
        tracing::info!(
            "FOCEI iteration {}: -2LL = {:.4} (change: {:.4})",
            self.iteration,
            self.objf,
            self.objf - self.prev_objf
        );

        tracing::debug!("Population mean (mu): {:?}", self.population.mu());
        tracing::debug!("Population SD: {:?}", self.population.standard_deviations());

        let pop_var = faer::Col::from_fn(self.population.npar(), |index| {
            self.population.omega()[(index, index)]
        });
        if let Some(shrinkage) = self.individual_estimates.shrinkage(&pop_var) {
            tracing::debug!("Shrinkage: {:?}", shrinkage);
        }
    }

    fn into_result(&self) -> Result<ParametricWorkspace<E>> {
        let likelihoods = LikelihoodEstimates {
            ll_linearization: Some(-self.objf / 2.0),
            ..LikelihoodEstimates::new()
        };

        assemble_focei_result(FoceiResultInput {
            equation: &self.equation,
            data: &self.data,
            population: &self.population,
            individual_estimates: &self.individual_estimates,
            objf: self.objf,
            iterations: self.iteration,
            status: &self.status,
            run_configuration: self.run_configuration.clone(),
            iteration_log: self.iteration_log.clone(),
            likelihood_estimates: likelihoods,
            transforms: &self.transforms,
            covariates: Some(covariate_state(
                self.subject_covariate_model.as_ref(),
                &self.subject_covariates,
                self.occasion_covariate_model.as_ref(),
                &self.occasion_covariates,
            )),
        })
    }
}
