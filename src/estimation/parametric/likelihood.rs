use anyhow::Result;
use faer::linalg::solvers::DenseSolveCore;
use faer::{Col, Mat};
use ndarray::Array2;
use pharmsol::{Data, Equation, Event, Predictions, ResidualErrorModels, Subject};

use crate::estimation::parametric::{
    phi_to_psi, ImportanceSamplingConfig, ImportanceSamplingEstimator, IndividualEstimates,
    LikelihoodEstimates, ParameterTransform, PhiVector, Population, SubjectConditionalPosterior,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualErrorUpdate {
    pub sigma_sq: f64,
    pub statrese: f64,
    pub n_observations: usize,
}

pub fn batch_log_likelihood_from_eta<E: Equation>(
    equation: &E,
    data: &Data,
    error_models: &ResidualErrorModels,
    transforms: &[ParameterTransform],
    eta_matrix: &Array2<f64>,
    mean_phi: &[Col<f64>],
) -> Result<Vec<f64>> {
    let n_subjects = eta_matrix.nrows();
    let n_params = eta_matrix.ncols();

    let mut psi_params = Array2::<f64>::zeros((n_subjects, n_params));
    for subject_index in 0..n_subjects {
        let phi = PhiVector(
            (0..n_params)
                .map(|param_index| {
                    mean_phi[subject_index][param_index] + eta_matrix[[subject_index, param_index]]
                })
                .collect(),
        );
        let psi = phi_to_psi(transforms, &phi);
        for (param_index, value) in psi.as_slice().iter().copied().enumerate() {
            psi_params[[subject_index, param_index]] = value;
        }
    }

    pharmsol::prelude::simulator::log_likelihood_batch(equation, data, &psi_params, error_models)
        .map_err(|error| anyhow::anyhow!("Likelihood computation failed: {}", error))
}

pub fn approximate_objective_from_individuals<E: Equation>(
    equation: &E,
    data: &Data,
    error_models: &ResidualErrorModels,
    transforms: &[ParameterTransform],
    population: &Population,
    individual_estimates: &IndividualEstimates,
    mean_phi: &[Col<f64>],
) -> f64 {
    let n_subjects = individual_estimates.nsubjects();
    if n_subjects == 0 {
        return f64::INFINITY;
    }
    let n_params = population.npar();

    let mut eta_matrix = Array2::<f64>::zeros((n_subjects, n_params));
    for subject_index in 0..n_subjects {
        if let Some(individual) = individual_estimates.get(subject_index) {
            for param_index in 0..n_params {
                eta_matrix[[subject_index, param_index]] = individual.eta()[param_index];
            }
        }
    }

    let log_likelihoods = match batch_log_likelihood_from_eta(
        equation,
        data,
        error_models,
        transforms,
        &eta_matrix,
        mean_phi,
    ) {
        Ok(log_likelihoods) => log_likelihoods,
        Err(_) => return f64::INFINITY,
    };

    let omega_inv = match population.omega().llt(faer::Side::Lower) {
        Ok(llt) => llt.inverse(),
        Err(_) => return f64::INFINITY,
    };

    let mut total_ll = 0.0;
    for (subject_index, log_likelihood) in log_likelihoods.iter().enumerate() {
        if !log_likelihood.is_finite() {
            continue;
        }

        let mut prior_term = 0.0;
        if let Some(individual) = individual_estimates.get(subject_index) {
            let eta = individual.eta();
            for row in 0..n_params {
                for col in 0..n_params {
                    prior_term += eta[row] * omega_inv[(row, col)] * eta[col];
                }
            }
        }

        total_ll += log_likelihood - 0.5 * prior_term;
    }

    -2.0 * total_ll
}

pub fn subject_objective_from_eta(
    subject_index: usize,
    eta_matrix: &Array2<f64>,
    log_likelihood: f64,
    omega_inv: &Mat<f64>,
) -> f64 {
    let quad = subject_eta_quadratic_form(subject_index, eta_matrix, omega_inv);

    -2.0 * (log_likelihood - 0.5 * quad)
}

pub fn subject_log_prior_from_eta(
    subject_index: usize,
    eta_matrix: &Array2<f64>,
    omega_inv: &Mat<f64>,
) -> f64 {
    -0.5 * subject_eta_quadratic_form(subject_index, eta_matrix, omega_inv)
}

pub fn log_priors_from_eta_matrix(eta_matrix: &Array2<f64>, omega_inv: &Mat<f64>) -> Vec<f64> {
    (0..eta_matrix.nrows())
        .map(|subject_index| subject_log_prior_from_eta(subject_index, eta_matrix, omega_inv))
        .collect()
}

fn subject_eta_quadratic_form(
    subject_index: usize,
    eta_matrix: &Array2<f64>,
    omega_inv: &Mat<f64>,
) -> f64 {
    let n_params = eta_matrix.ncols();
    (0..n_params)
        .flat_map(|row| {
            (0..n_params).map(move |col| {
                eta_matrix[[subject_index, row]]
                    * omega_inv[(row, col)]
                    * eta_matrix[[subject_index, col]]
            })
        })
        .sum::<f64>()
}

pub fn estimate_initial_sigma_sq(_error_models: &ResidualErrorModels) -> f64 {
    1.0
}

pub fn sync_error_models_with_sigma(error_models: &mut ResidualErrorModels, sigma_sq: f64) {
    error_models.update_sigma(sigma_sq.sqrt());
}

pub fn update_residual_error_from_individuals<E: Equation>(
    equation: &E,
    data: &Data,
    error_models: &mut ResidualErrorModels,
    transforms: &[ParameterTransform],
    individual_estimates: &IndividualEstimates,
    step_size: f64,
    sigma_sq: f64,
    statrese: f64,
    use_annealed_sigma_floor: bool,
    sa_alpha: f64,
    allow_sigma_update: bool,
) -> Result<ResidualErrorUpdate> {
    let mut sum_weighted_sq_residuals = 0.0;
    let mut n_observations = 0;

    for (subject_index, subject) in data.subjects().iter().enumerate() {
        let Some(individual) = individual_estimates.get(subject_index) else {
            continue;
        };

        let phi = PhiVector::from(individual.psi());
        let params = phi_to_psi(transforms, &phi).0;

        let Ok(predictions) = equation.estimate_predictions(subject, &params) else {
            continue;
        };

        let observations: Vec<_> = subject
            .occasions()
            .iter()
            .flat_map(|occasion| occasion.events().iter())
            .filter_map(|event| {
                if let Event::Observation(observation) = event {
                    observation
                        .value()
                        .map(|value| (value, observation.outeq()))
                } else {
                    None
                }
            })
            .collect();

        for ((observation_value, outeq), prediction) in observations
            .iter()
            .zip(predictions.get_predictions().iter())
        {
            let predicted_value = prediction.prediction();
            if let Some(error_model) = error_models.get(*outeq) {
                sum_weighted_sq_residuals +=
                    error_model.weighted_squared_residual(*observation_value, predicted_value);
            } else {
                let residual = observation_value - predicted_value;
                sum_weighted_sq_residuals += residual * residual;
            }
            n_observations += 1;
        }
    }

    if n_observations == 0 {
        return Ok(ResidualErrorUpdate {
            sigma_sq,
            statrese,
            n_observations,
        });
    }

    let updated_statrese = if step_size > 0.0 {
        statrese + step_size * (sum_weighted_sq_residuals - statrese)
    } else {
        statrese
    };
    let sig2 = updated_statrese / n_observations as f64;
    let updated_sigma_sq = if use_annealed_sigma_floor {
        let decayed_sigma = sigma_sq.sqrt() * sa_alpha;
        decayed_sigma.max(sig2.sqrt()).powi(2)
    } else if allow_sigma_update {
        sig2
    } else {
        sigma_sq
    };

    sync_error_models_with_sigma(error_models, updated_sigma_sq);

    Ok(ResidualErrorUpdate {
        sigma_sq: updated_sigma_sq,
        statrese: updated_statrese,
        n_observations,
    })
}

pub fn importance_sampling_likelihood_estimates<E: Equation>(
    equation: &E,
    subjects: Vec<&Subject>,
    error_models: &ResidualErrorModels,
    transforms: &[ParameterTransform],
    mu_phi: &Col<f64>,
    omega: &Mat<f64>,
    conditionals: &[SubjectConditionalPosterior],
    config: ImportanceSamplingConfig,
) -> Result<LikelihoodEstimates> {
    let estimator = ImportanceSamplingEstimator::new(
        config.clone(),
        equation,
        error_models,
        transforms,
        mu_phi,
        omega,
    )?;
    let minus2ll = estimator.estimate_minus2ll(subjects, conditionals);

    let mut estimates = LikelihoodEstimates::new();
    if minus2ll.is_finite() {
        estimates.ll_importance_sampling = Some(-minus2ll / 2.0);
        estimates.is_n_samples = Some(config.n_samples);
    }

    Ok(estimates)
}

pub fn subject_conditionals_from_eta_samples(
    eta_samples_by_subject: &[Vec<Col<f64>>],
    fallback_individuals: Option<&IndividualEstimates>,
    mu_phi: &Col<f64>,
    omega: &Mat<f64>,
) -> Vec<SubjectConditionalPosterior> {
    eta_samples_by_subject
        .iter()
        .enumerate()
        .map(|(index, eta_samples)| {
            if eta_samples.is_empty() {
                fallback_subject_conditional(fallback_individuals, index, mu_phi, omega)
            } else {
                SubjectConditionalPosterior::from_mcmc_samples(eta_samples, mu_phi, omega)
            }
        })
        .collect()
}

fn fallback_subject_conditional(
    fallback_individuals: Option<&IndividualEstimates>,
    index: usize,
    mu_phi: &Col<f64>,
    omega: &Mat<f64>,
) -> SubjectConditionalPosterior {
    let mean = fallback_individuals
        .and_then(|individuals| individuals.get(index))
        .map(|individual| individual.psi().clone())
        .unwrap_or_else(|| mu_phi.clone());
    let variance = (0..mu_phi.nrows())
        .map(|parameter_index| omega[(parameter_index, parameter_index)].max(1e-6))
        .collect();

    SubjectConditionalPosterior::new(mean, variance)
}

#[cfg(test)]
mod tests {
    use faer::{Col, Mat};

    use super::subject_conditionals_from_eta_samples;
    use crate::estimation::parametric::{Individual, IndividualEstimates};

    #[test]
    fn falls_back_to_individual_phi_when_no_chain_samples_exist() {
        let mu_phi = Col::from_fn(2, |index| if index == 0 { 0.0 } else { 1.0 });
        let omega = Mat::from_fn(2, 2, |row, col| if row == col { 0.25 } else { 0.0 });
        let individual = Individual::new(
            "1",
            Col::from_fn(2, |_| 0.0),
            Col::from_fn(2, |index| if index == 0 { 0.2 } else { 1.3 }),
        )
        .expect("valid individual");
        let individuals = IndividualEstimates::from_vec(vec![individual]);

        let conditionals = subject_conditionals_from_eta_samples(
            &[Vec::new()],
            Some(&individuals),
            &mu_phi,
            &omega,
        );

        assert_eq!(conditionals.len(), 1);
        assert!((conditionals[0].mean[0] - 0.2).abs() < 1e-12);
        assert!((conditionals[0].mean[1] - 1.3).abs() < 1e-12);
        assert_eq!(conditionals[0].variance, vec![0.25, 0.25]);
    }
}
