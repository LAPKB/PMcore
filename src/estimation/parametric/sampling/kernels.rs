//! Core MCMC types for SAEM algorithm.

use faer::Col;
use faer::Mat;
use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::estimation::parametric::Population;

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
}
