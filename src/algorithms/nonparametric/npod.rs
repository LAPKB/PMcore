use anyhow::Result;
use ndarray::Array1;
use pharmsol::prelude::{data::AssayErrorModels, simulator::Equation};
use pharmsol::ParameterOptimizer;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::error_optim::{ErrorModelOptimizer, ErrorOptimConfig};
use super::fit::{CycleOutcome, FitState, Runner};
use crate::estimation::nonparametric::Theta;

/// Configuration options for the Non-Parametric Optimal Design (NPOD) algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NpodConfig {
    /// Maximum number of cycles to run the algorithm for.
    pub max_cycles: usize,
    /// Converged once the log-likelihood changes by at most this much in a cycle.
    pub objective_tolerance: f64,
    /// Minimum distance between a new candidate point and the existing support points.
    pub min_distance: f64,
    /// Support points with at most this fraction of the largest weight are dropped.
    pub prune_threshold: f64,
    /// Tolerance of the rank-revealing QR decomposition that drops linearly dependent points.
    pub qr_tolerance: f64,
    /// Configuration for the error-model factor (gamma/lambda) optimization.
    pub error_optim: ErrorOptimConfig,
    /// Whether to print progress information during the first cycle.
    pub progress: bool,
}

impl NpodConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_cycles(mut self, cycles: usize) -> Self {
        self.max_cycles = cycles;
        self
    }

    pub fn objective_tolerance(mut self, tolerance: f64) -> Self {
        self.objective_tolerance = tolerance;
        self
    }

    pub fn min_distance(mut self, distance: f64) -> Self {
        self.min_distance = distance;
        self
    }

    pub fn prune_threshold(mut self, threshold: f64) -> Self {
        self.prune_threshold = threshold;
        self
    }

    pub fn qr_tolerance(mut self, tolerance: f64) -> Self {
        self.qr_tolerance = tolerance;
        self
    }

    pub fn error_optim(mut self, config: ErrorOptimConfig) -> Self {
        self.error_optim = config;
        self
    }

    pub fn progress(mut self, progress: bool) -> Self {
        self.progress = progress;
        self
    }
}

impl Default for NpodConfig {
    fn default() -> Self {
        Self {
            max_cycles: 100,
            objective_tolerance: 1e-2,
            min_distance: 1e-4,
            prune_threshold: 1e-3,
            qr_tolerance: 1e-8,
            error_optim: ErrorOptimConfig::default(),
            progress: true,
        }
    }
}

/// NPOD's own state between cycles: its configuration and the error-model optimizer.
#[derive(Debug)]
pub(crate) struct Npod {
    config: NpodConfig,
    error_optimizer: ErrorModelOptimizer,
}

impl Npod {
    pub(crate) fn new(config: NpodConfig, error_models: &AssayErrorModels) -> Self {
        Self {
            error_optimizer: ErrorModelOptimizer::new(config.error_optim.clone(), error_models),
            config,
        }
    }
}

impl<E: Equation> Runner<E> for Npod {
    fn step(&mut self, state: &mut FitState<E>) -> Result<CycleOutcome> {
        let theta = if state.is_first_cycle() {
            state.solution().theta().clone()
        } else {
            expand(state, self.config.min_distance)?
        };

        let estimate = state.estimate(theta, self.config.progress && state.is_first_cycle())?;
        let solution = state.condense(
            estimate,
            self.config.prune_threshold,
            self.config.qr_tolerance,
        )?;
        state.set_solution(solution);
        self.error_optimizer.optimize(state)?;

        let converged = state
            .objective_change()
            .is_some_and(|change| change <= self.config.objective_tolerance);
        Ok(if converged {
            CycleOutcome::Converged
        } else {
            CycleOutcome::Continue
        })
    }
}

/// Move each support point towards a local maximum of the directional derivative of the
/// likelihood, and return the support points with the results added as new candidates.
fn expand<E: Equation>(state: &FitState<E>, min_distance: f64) -> Result<Theta> {
    let solution = state.solution();
    // The population likelihood of each subject, Ψ·w. `ParameterOptimizer` works with raw
    // likelihoods, so undo the per-subject scaling of Ψ.
    // TODO: Move NPOD and its parameter optimizer fully into log space.
    let pyl: Array1<f64> = (solution.psi().matrix() * solution.weights().weights())
        .iter()
        .zip(solution.psi().row_log_scales())
        .map(|(pyl, row_log_scale)| pyl * row_log_scale.exp())
        .collect();

    let candidates: Vec<Array1<f64>> = solution
        .theta()
        .matrix()
        .row_iter()
        .map(|row| row.iter().copied().collect())
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|point| {
            ParameterOptimizer::new(state.equation(), state.data(), state.error_models(), &pyl)
                .optimize_point(point)
        })
        .collect::<Result<_, _>>()?;

    let mut theta = solution.theta().clone();
    for candidate in candidates {
        theta.suggest_point(&candidate.to_vec(), min_distance)?;
    }
    Ok(theta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_without_the_newer_fields_still_deserializes() -> anyhow::Result<()> {
        let config: NpodConfig = serde_json::from_str(r#"{"max_cycles": 7, "progress": false}"#)?;

        let defaults = NpodConfig::default();
        assert_eq!(config.max_cycles, 7);
        assert!(!config.progress);
        assert_eq!(config.objective_tolerance, defaults.objective_tolerance);
        assert_eq!(config.min_distance, defaults.min_distance);
        assert_eq!(config.prune_threshold, defaults.prune_threshold);
        assert_eq!(config.qr_tolerance, defaults.qr_tolerance);
        Ok(())
    }
}
