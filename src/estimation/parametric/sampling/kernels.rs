//! Core MCMC types for SAEM algorithm.

use anyhow::Result;
use faer::Col;
use faer::Mat;
use ndarray::Array2;
use pharmsol::{Data, Equation, ResidualErrorModels};
use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::estimation::parametric::{
    batch_log_likelihood_from_eta, log_priors_from_eta_matrix, ParameterTransform, Population,
};

#[derive(Debug, Clone)]
pub struct KernelConfig {
    pub n_kernel1: usize,
    pub n_kernel2: usize,
    pub n_kernel3: usize,
    pub n_kernel4: usize,
    pub map_iterations: usize,
    pub rw_step_size: f64,
    pub target_acceptance: f64,
    pub rw_init: f64,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            n_kernel1: 2,
            n_kernel2: 2,
            n_kernel3: 2,
            n_kernel4: 0,
            map_iterations: 0,
            rw_step_size: 0.4,
            target_acceptance: 0.4,
            rw_init: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChainState {
    pub eta: Col<f64>,
    pub log_likelihood: f64,
    pub log_prior: f64,
}

impl Default for ChainState {
    fn default() -> Self {
        Self {
            eta: Col::zeros(0),
            log_likelihood: f64::NEG_INFINITY,
            log_prior: f64::NEG_INFINITY,
        }
    }
}

impl ChainState {
    pub fn new(eta: Col<f64>) -> Self {
        Self {
            eta,
            log_likelihood: f64::NEG_INFINITY,
            log_prior: f64::NEG_INFINITY,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SaemMcmcState {
    pub eta_matrix: Array2<f64>,
    pub log_likelihoods: Vec<f64>,
    pub log_priors: Vec<f64>,
}

pub(crate) fn advance_saem_chains<E: Equation>(
    equation: &E,
    data: &Data,
    error_models: &ResidualErrorModels,
    transforms: &[ParameterTransform],
    mean_phi: &[Col<f64>],
    chol_omega: &Mat<f64>,
    omega_inv: &Mat<f64>,
    kernel_config: &KernelConfig,
    iteration: usize,
    domega2: &mut Col<f64>,
    rng: &mut impl Rng,
    mut eta_matrix: Array2<f64>,
) -> Result<SaemMcmcState> {
    let n_subjects = eta_matrix.nrows();
    let n_params = eta_matrix.ncols();
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut current_ll = batch_log_likelihood_from_eta(
        equation,
        data,
        error_models,
        transforms,
        &eta_matrix,
        mean_phi,
    )?;
    let mut current_log_prior = log_priors_from_eta_matrix(&eta_matrix, omega_inv);

    for _ in 0..kernel_config.n_kernel1 {
        let proposed_eta = prior_proposals(chol_omega, n_subjects, rng, &normal);
        let proposed_ll = batch_log_likelihood_from_eta(
            equation,
            data,
            error_models,
            transforms,
            &proposed_eta,
            mean_phi,
        )?;

        for subject_index in 0..n_subjects {
            let log_alpha = proposed_ll[subject_index] - current_ll[subject_index];
            let u: f64 = rng.random();
            if log_alpha.is_finite() && u.ln() < log_alpha {
                for param_index in 0..n_params {
                    eta_matrix[[subject_index, param_index]] =
                        proposed_eta[[subject_index, param_index]];
                }
                current_ll[subject_index] = proposed_ll[subject_index];
                current_log_prior[subject_index] =
                    subject_log_prior(subject_index, &eta_matrix, omega_inv);
            }
        }
    }

    if kernel_config.n_kernel2 > 0 {
        let mut accepted = vec![0usize; n_params];
        let mut total = vec![0usize; n_params];

        for _ in 0..kernel_config.n_kernel2 {
            for param_index in 0..n_params {
                let mut proposed_eta = eta_matrix.clone();
                for subject_index in 0..n_subjects {
                    let perturbation = normal.sample(rng) * domega2[param_index];
                    proposed_eta[[subject_index, param_index]] += perturbation;
                }

                let proposed_ll = batch_log_likelihood_from_eta(
                    equation,
                    data,
                    error_models,
                    transforms,
                    &proposed_eta,
                    mean_phi,
                )?;
                let proposed_log_prior = log_priors_from_eta_matrix(&proposed_eta, omega_inv);

                for subject_index in 0..n_subjects {
                    let log_alpha = (proposed_ll[subject_index] + proposed_log_prior[subject_index])
                        - (current_ll[subject_index] + current_log_prior[subject_index]);
                    let u: f64 = rng.random();
                    if log_alpha.is_finite() && u.ln() < log_alpha {
                        eta_matrix[[subject_index, param_index]] =
                            proposed_eta[[subject_index, param_index]];
                        current_ll[subject_index] = proposed_ll[subject_index];
                        current_log_prior[subject_index] = proposed_log_prior[subject_index];
                        accepted[param_index] += 1;
                    }
                    total[param_index] += 1;
                }
            }
        }

        adapt_proposal_scales(domega2, &accepted, &total, kernel_config);
    }

    if kernel_config.n_kernel3 > 0 {
        let mut accepted = vec![0usize; n_params];
        let mut total = vec![0usize; n_params];

        for _ in 0..kernel_config.n_kernel3 {
            let block_indices = block_indices_for_iteration(iteration, n_params, rng);
            let mut proposed_eta = eta_matrix.clone();

            for subject_index in 0..n_subjects {
                for &param_index in &block_indices {
                    let perturbation = normal.sample(rng) * domega2[param_index];
                    proposed_eta[[subject_index, param_index]] += perturbation;
                }
            }

            let proposed_ll = batch_log_likelihood_from_eta(
                equation,
                data,
                error_models,
                transforms,
                &proposed_eta,
                mean_phi,
            )?;
            let proposed_log_prior = log_priors_from_eta_matrix(&proposed_eta, omega_inv);

            for subject_index in 0..n_subjects {
                let log_alpha = (proposed_ll[subject_index] + proposed_log_prior[subject_index])
                    - (current_ll[subject_index] + current_log_prior[subject_index]);
                let u: f64 = rng.random();
                if log_alpha.is_finite() && u.ln() < log_alpha {
                    for &param_index in &block_indices {
                        eta_matrix[[subject_index, param_index]] =
                            proposed_eta[[subject_index, param_index]];
                        accepted[param_index] += 1;
                    }
                    current_ll[subject_index] = proposed_ll[subject_index];
                    current_log_prior[subject_index] = proposed_log_prior[subject_index];
                }
                for &param_index in &block_indices {
                    total[param_index] += 1;
                }
            }
        }

        adapt_proposal_scales(domega2, &accepted, &total, kernel_config);
    }

    Ok(SaemMcmcState {
        eta_matrix,
        log_likelihoods: current_ll,
        log_priors: current_log_prior,
    })
}

pub(crate) fn sample_eta_from_population(population: &Population, rng: &mut impl Rng) -> Col<f64> {
    let n = population.npar();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let z: Vec<f64> = (0..n).map(|_| normal.sample(rng)).collect();

    let omega = population.omega();
    let chol = match omega.llt(faer::Side::Lower) {
        Ok(llt) => llt.L().to_owned(),
        Err(_) => {
            let mut diagonal = Mat::zeros(n, n);
            for i in 0..n {
                diagonal[(i, i)] = omega[(i, i)].sqrt().max(1e-6);
            }
            diagonal
        }
    };

    let mut eta = Col::zeros(n);
    for i in 0..n {
        for j in 0..=i {
            eta[i] += chol[(i, j)] * z[j];
        }
    }

    eta
}

fn prior_proposals(
    chol_omega: &Mat<f64>,
    n_subjects: usize,
    rng: &mut impl Rng,
    normal: &Normal<f64>,
) -> Array2<f64> {
    let n_params = chol_omega.nrows();
    let mut proposed_eta = Array2::zeros((n_subjects, n_params));

    for subject_index in 0..n_subjects {
        let z: Vec<f64> = (0..n_params).map(|_| normal.sample(rng)).collect();
        for row in 0..n_params {
            let mut sum = 0.0;
            for col in 0..=row {
                sum += chol_omega[(row, col)] * z[col];
            }
            proposed_eta[[subject_index, row]] = sum;
        }
    }

    proposed_eta
}

fn subject_log_prior(subject_index: usize, eta_matrix: &Array2<f64>, omega_inv: &Mat<f64>) -> f64 {
    let n_params = eta_matrix.ncols();
    let quadratic = (0..n_params)
        .flat_map(|row| {
            (0..n_params).map(move |col| {
                eta_matrix[[subject_index, row]]
                    * omega_inv[(row, col)]
                    * eta_matrix[[subject_index, col]]
            })
        })
        .sum::<f64>();

    -0.5 * quadratic
}

fn block_indices_for_iteration(
    iteration: usize,
    n_params: usize,
    rng: &mut impl Rng,
) -> Vec<usize> {
    let block_size = ((iteration % n_params.max(2).saturating_sub(1)).max(1) + 1).min(n_params);
    if block_size >= n_params {
        return (0..n_params).collect();
    }

    let mut indices: Vec<usize> = (0..n_params).collect();
    for offset in 0..block_size {
        let u: f64 = rng.random();
        let remaining = n_params - offset;
        let swap_offset = (u * remaining as f64).floor() as usize;
        let swap_index = offset + swap_offset.min(remaining - 1);
        indices.swap(offset, swap_index);
    }

    indices[..block_size].to_vec()
}

fn adapt_proposal_scales(
    domega2: &mut Col<f64>,
    accepted: &[usize],
    total: &[usize],
    kernel_config: &KernelConfig,
) {
    for param_index in 0..domega2.nrows() {
        if total[param_index] > 0 {
            let acc_rate = accepted[param_index] as f64 / total[param_index] as f64;
            domega2[param_index] *= 1.0
                + kernel_config.rw_step_size * (acc_rate - kernel_config.target_acceptance);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::parametric::Population;
    use crate::model::{ParameterSpace, ParameterSpec};
    use faer::Mat;

    #[test]
    fn test_chain_state() {
        let eta = Col::from_fn(3, |i| i as f64 * 0.1);
        let state = ChainState::new(eta);
        assert_eq!(state.log_likelihood, f64::NEG_INFINITY);
        assert_eq!(state.log_prior, f64::NEG_INFINITY);
    }

    #[test]
    fn test_sample_eta_from_population_matches_dimension() {
        let parameters = ParameterSpace::new()
            .add(ParameterSpec::bounded("ke", 0.1, 1.0))
            .add(ParameterSpec::bounded("v", 1.0, 20.0));
        let population = Population::new(
            Col::from_fn(2, |_| 0.0),
            Mat::from_fn(2, 2, |row, col| if row == col { 0.5 } else { 0.0 }),
            parameters,
        )
        .unwrap();
        let mut rng = rand::rng();

        let eta = sample_eta_from_population(&population, &mut rng);

        assert_eq!(eta.nrows(), 2);
    }

    #[test]
    fn test_adapt_proposal_scales_uses_acceptance_gap() {
        let mut domega2 = Col::from_fn(2, |_| 1.0);
        let config = KernelConfig {
            rw_step_size: 0.5,
            target_acceptance: 0.4,
            ..KernelConfig::default()
        };

        adapt_proposal_scales(&mut domega2, &[8, 1], &[10, 10], &config);

        assert!(domega2[0] > 1.0);
        assert!(domega2[1] < 1.0);
    }

    #[test]
    fn test_block_indices_respect_parameter_count() {
        let mut rng = rand::rng();

        let indices = block_indices_for_iteration(5, 4, &mut rng);

        assert!(!indices.is_empty());
        assert!(indices.len() <= 4);
        for index in &indices {
            assert!(*index < 4);
        }
    }
}
