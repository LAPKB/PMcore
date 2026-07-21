use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};

use argmin::core::{CostFunction, Error, Executor};
use argmin::solver::neldermead::NelderMead;
use pharmsol::{Data, Parameters, SDE};

use super::DiffusionConfig;
use crate::{AssayErrorModels, SdeParticleConfig, SdeParticleFilter};

const INVALID_CANDIDATE_COST: f64 = 1e10;

pub(crate) struct SigmaCost<'a> {
    sde: &'a SDE,
    data: &'a Data,
    primary: Vec<f64>,
    primary_indices: Vec<usize>,
    sigma_indices: Vec<usize>,
    parameter_names: Vec<String>,
    error_models: &'a AssayErrorModels,
    responsibilities: Option<&'a [f64]>,
    n_total: usize,
    particle_count: usize,
    seed: AtomicU64,
    execution_error: Mutex<Option<String>>,
}

impl<'a> SigmaCost<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        sde: &'a SDE,
        data: &'a Data,
        primary: &[f64],
        primary_indices: &[usize],
        sigma_indices: &[usize],
        parameter_names: Vec<String>,
        error_models: &'a AssayErrorModels,
        responsibilities: Option<&'a [f64]>,
        particle_count: usize,
    ) -> Self {
        Self {
            sde,
            data,
            primary: primary.to_vec(),
            primary_indices: primary_indices.to_vec(),
            sigma_indices: sigma_indices.to_vec(),
            parameter_names,
            error_models,
            responsibilities,
            n_total: primary.len() + sigma_indices.len(),
            particle_count,
            seed: AtomicU64::new(0),
            execution_error: Mutex::new(None),
        }
    }

    fn build_params(&self, sigma: &[f64]) -> Vec<f64> {
        let mut full = vec![0.0; self.n_total];
        for (&index, &value) in self.primary_indices.iter().zip(&self.primary) {
            full[index] = value;
        }
        for (&index, &value) in self.sigma_indices.iter().zip(sigma) {
            full[index] = value;
        }
        full
    }

    fn record_execution_error(&self, error: String) {
        let mut slot = self
            .execution_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if slot.is_none() {
            *slot = Some(error);
        }
    }

    fn take_execution_error(&self) -> Option<String> {
        self.execution_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }
}

impl CostFunction for SigmaCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, sigma: &Self::Param) -> Result<Self::Output, Error> {
        if sigma.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return Ok(1e10);
        }
        let full = self.build_params(sigma);
        let parameters = match Parameters::with_model(
            self.sde,
            self.parameter_names
                .iter()
                .map(String::as_str)
                .zip(full.iter().copied()),
        ) {
            Ok(parameters) => parameters,
            Err(error) => {
                self.record_execution_error(error.to_string());
                return Ok(INVALID_CANDIDATE_COST);
            }
        };
        let call = self.seed.fetch_add(1, Ordering::Relaxed);
        let config = SdeParticleConfig::new(self.particle_count)
            .with_process_seed(call.wrapping_mul(2))
            .with_resampling_seed(call.wrapping_mul(2).wrapping_add(1));

        let mut total = 0.0;
        for (subject_index, subject) in self.data.subjects().iter().enumerate() {
            let result =
                match self
                    .sde
                    .particle_filter(subject, &parameters, self.error_models, &config)
                {
                    Ok(result) => result,
                    Err(error) => {
                        self.record_execution_error(error.to_string());
                        return Ok(INVALID_CANDIDATE_COST);
                    }
                };
            let responsibility = self
                .responsibilities
                .map_or(1.0, |values| values[subject_index]);
            total += responsibility * result.log_value;
        }
        if !total.is_finite() {
            self.record_execution_error("particle-filter objective is non-finite".to_string());
            return Ok(INVALID_CANDIDATE_COST);
        }
        Ok(-total)
    }
}

struct ResampledCost<'a> {
    inner: &'a SigmaCost<'a>,
    samples: usize,
    bounds: &'a [(f64, f64)],
}

impl CostFunction for ResampledCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, sigma: &Self::Param) -> Result<Self::Output, Error> {
        if !candidate_within_bounds(sigma, self.bounds) {
            return Ok(INVALID_CANDIDATE_COST);
        }

        let mut sum = 0.0;
        for _ in 0..self.samples.max(1) {
            sum += self.inner.cost(sigma)?;
        }
        Ok(sum / self.samples.max(1) as f64)
    }
}

fn candidate_within_bounds(candidate: &[f64], bounds: &[(f64, f64)]) -> bool {
    candidate.len() == bounds.len()
        && candidate
            .iter()
            .zip(bounds)
            .all(|(&value, &(lower, upper))| value.is_finite() && value >= lower && value <= upper)
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub(crate) enum DiffusionOptimizationError {
    #[error("failed to configure diffusion optimizer: {0}")]
    SolverConfiguration(String),
    #[error("diffusion optimizer execution failed: {0}")]
    Execution(String),
    #[error("diffusion optimizer did not return a best parameter")]
    MissingBestParameter,
    #[error("diffusion optimizer returned non-finite final cost {0}")]
    NonFiniteFinalCost(f64),
    #[error("optimizer returned {actual} diffusion parameters, expected {expected}")]
    ReturnedDimension { expected: usize, actual: usize },
    #[error(
        "optimizer returned invalid diffusion parameter at position {position}: {value} is not finite or outside inclusive bounds [{lower}, {upper}]"
    )]
    ReturnedParameter {
        position: usize,
        value: f64,
        lower: f64,
        upper: f64,
    },
}

fn validate_returned_params(
    candidate: &[f64],
    bounds: &[(f64, f64)],
) -> Result<(), DiffusionOptimizationError> {
    if candidate.len() != bounds.len() {
        return Err(DiffusionOptimizationError::ReturnedDimension {
            expected: bounds.len(),
            actual: candidate.len(),
        });
    }
    for (position, (&value, &(lower, upper))) in candidate.iter().zip(bounds).enumerate() {
        if !value.is_finite() || value < lower || value > upper {
            return Err(DiffusionOptimizationError::ReturnedParameter {
                position,
                value,
                lower,
                upper,
            });
        }
    }
    Ok(())
}

pub(crate) struct OptimizationOutcome {
    pub optimized_params: Vec<f64>,
    pub final_cost: f64,
    pub iterations: usize,
    pub converged: bool,
}

pub(crate) fn optimize_sigma(
    cost: SigmaCost<'_>,
    sigma_init: &[f64],
    sigma_bounds: &[(f64, f64)],
    config: &DiffusionConfig,
) -> Result<OptimizationOutcome, DiffusionOptimizationError> {
    let simplex = build_simplex(sigma_init, sigma_bounds, config.initial_perturbation);
    let resampled = ResampledCost {
        inner: &cost,
        samples: config.resampling_samples,
        bounds: sigma_bounds,
    };
    let solver: NelderMead<Vec<f64>, f64> = NelderMead::new(simplex)
        .with_sd_tolerance(config.sd_tolerance)
        .map_err(|error| DiffusionOptimizationError::SolverConfiguration(error.to_string()))?;
    let execution = Executor::new(resampled, solver)
        .configure(|state| state.max_iters(config.max_iter as u64))
        .run();
    if let Some(error) = cost.take_execution_error() {
        return Err(DiffusionOptimizationError::Execution(error));
    }
    let result =
        execution.map_err(|error| DiffusionOptimizationError::Execution(error.to_string()))?;
    let iterations = result.state.iter as usize;
    let optimized_params = result
        .state
        .best_param
        .ok_or(DiffusionOptimizationError::MissingBestParameter)?;
    let final_cost = result.state.best_cost;
    if !final_cost.is_finite() {
        return Err(DiffusionOptimizationError::NonFiniteFinalCost(final_cost));
    }
    validate_returned_params(&optimized_params, sigma_bounds)?;

    Ok(OptimizationOutcome {
        optimized_params,
        final_cost,
        iterations,
        converged: iterations < config.max_iter,
    })
}

fn build_simplex(initial: &[f64], bounds: &[(f64, f64)], perturbation: f64) -> Vec<Vec<f64>> {
    let mut vertices = Vec::with_capacity(initial.len() + 1);
    vertices.push(initial.to_vec());
    for index in 0..initial.len() {
        let mut vertex = initial.to_vec();
        let (lower, upper) = bounds[index];
        vertex[index] =
            (initial[index] + perturbation * (upper - initial[index])).clamp(lower, upper);
        if (vertex[index] - initial[index]).abs() < 1e-10 {
            vertex[index] = (lower + upper) * 0.5;
        }
        vertices.push(vertex);
    }
    vertices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplex_reaches_optimum() {
        let simplex = build_simplex(&[0.01], &[(0.0, 0.5)], 0.15);
        assert!((simplex[1][0] - 0.0835).abs() < 1e-10);
        let simplex = build_simplex(&[0.3], &[(0.0, 1.0)], 0.15);
        assert!((simplex[1][0] - 0.405).abs() < 1e-10);
    }

    #[test]
    fn candidate_bounds_enforce_positive_lower_and_upper_limits() {
        let bounds = [(0.25, 0.75), (1.0, 2.0)];

        assert!(candidate_within_bounds(&[0.25, 2.0], &bounds));
        assert!(!candidate_within_bounds(&[0.249, 1.5], &bounds));
        assert!(!candidate_within_bounds(&[0.5, 2.001], &bounds));
        assert!(!candidate_within_bounds(&[f64::NAN, 1.5], &bounds));
        assert!(!candidate_within_bounds(&[0.5], &bounds));
    }

    #[test]
    fn returned_point_validation_rejects_each_bound_violation() {
        let bounds = [(0.25, 0.75), (1.0, 2.0)];

        assert!(validate_returned_params(&[0.25, 2.0], &bounds).is_ok());
        assert!(matches!(
            validate_returned_params(&[0.249, 1.5], &bounds),
            Err(DiffusionOptimizationError::ReturnedParameter { position: 0, .. })
        ));
        assert!(matches!(
            validate_returned_params(&[0.5, 2.001], &bounds),
            Err(DiffusionOptimizationError::ReturnedParameter { position: 1, .. })
        ));
        assert!(matches!(
            validate_returned_params(&[0.5, f64::INFINITY], &bounds),
            Err(DiffusionOptimizationError::ReturnedParameter { position: 1, .. })
        ));
    }
}
